//! JSON Schema Draft 2020-12 validators for every entity written via the
//! Admin API (spec §2, §3).
//!
//! The flow on write is:
//! ```text
//! 1. parse bytes as serde_json::Value
//! 2. validator.validate(&value)       → emits detailed field path on failure
//! 3. serde deserialise into the typed struct (cheap after schema passes)
//! 4. duplicate-name check vs snapshot
//! 5. etcd txn commit
//! ```
//!
//! The watch path reuses step 2 on incoming events — malformed payloads are
//! skipped with a warning and do not take down the gateway.

use jsonschema::Validator;
use once_cell::sync::Lazy;
use serde_json::{json, Value};
use std::sync::Arc;
use thiserror::Error;

/// Cached compiled schemas. Compiling on every write would be wasteful; the
/// schemas are static, so we build them once.
pub struct Schemas {
    pub model: Validator,
    pub apikey: Validator,
    pub provider_key: Validator,
    pub guardrail: Validator,
    pub guardrail_attachment: Validator,
    pub cache_policy: Validator,
    pub observability_exporter: Validator,
    pub rate_limit_policy: Validator,
    pub mcp_server: Validator,
    pub a2a_agent: Validator,
}

pub static SCHEMAS: Lazy<Arc<Schemas>> = Lazy::new(|| Arc::new(Schemas::compile()));

impl Schemas {
    fn compile() -> Self {
        Self {
            model: jsonschema::options()
                .build(&model_root_schema())
                .expect("model schema is well-formed"),
            apikey: jsonschema::options()
                .build(&apikey_root_schema())
                .expect("apikey schema is well-formed"),
            provider_key: jsonschema::options()
                .build(&provider_key_root_schema())
                .expect("provider_key schema is well-formed"),
            guardrail: jsonschema::options()
                .build(&guardrail_root_schema())
                .expect("guardrail schema is well-formed"),
            guardrail_attachment: jsonschema::options()
                .build(&guardrail_attachment_root_schema())
                .expect("guardrail_attachment schema is well-formed"),
            cache_policy: jsonschema::options()
                .build(&cache_policy_root_schema())
                .expect("cache_policy schema is well-formed"),
            observability_exporter: jsonschema::options()
                .build(&observability_exporter_root_schema())
                .expect("observability_exporter schema is well-formed"),
            rate_limit_policy: jsonschema::options()
                .build(&rate_limit_policy_root_schema())
                .expect("rate_limit_policy schema is well-formed"),
            mcp_server: jsonschema::options()
                .build(&mcp_server_root_schema())
                .expect("mcp_server schema is well-formed"),
            a2a_agent: jsonschema::options()
                .build(&a2a_agent_root_schema())
                .expect("a2a_agent schema is well-formed"),
        }
    }
}

#[derive(Debug, Error)]
#[error("schema validation failed at `{path}`: {message}")]
pub struct SchemaError {
    pub path: String,
    pub message: String,
}

/// Run a compiled validator and collapse all errors into a single
/// human-readable message containing the first failing JSON pointer.
pub fn validate(validator: &Validator, value: &Value) -> Result<(), SchemaError> {
    let mut errors = validator.iter_errors(value);
    if let Some(err) = errors.next() {
        return Err(SchemaError {
            path: err.instance_path.to_string(),
            message: err.to_string(),
        });
    }
    Ok(())
}

pub fn validate_model(value: &Value) -> Result<(), SchemaError> {
    validate(&SCHEMAS.model, value)
}

pub fn validate_apikey(value: &Value) -> Result<(), SchemaError> {
    validate(&SCHEMAS.apikey, value)
}

pub fn validate_provider_key(value: &Value) -> Result<(), SchemaError> {
    validate(&SCHEMAS.provider_key, value)
}

pub fn validate_guardrail(value: &Value) -> Result<(), SchemaError> {
    validate(&SCHEMAS.guardrail, value)
}

pub fn validate_cache_policy(value: &Value) -> Result<(), SchemaError> {
    validate(&SCHEMAS.cache_policy, value)
}

pub fn validate_observability_exporter(value: &Value) -> Result<(), SchemaError> {
    validate(&SCHEMAS.observability_exporter, value)
}

pub fn validate_rate_limit_policy(value: &Value) -> Result<(), SchemaError> {
    validate(&SCHEMAS.rate_limit_policy, value)
}

pub fn validate_guardrail_attachment(value: &Value) -> Result<(), SchemaError> {
    validate(&SCHEMAS.guardrail_attachment, value)
}

pub fn validate_mcp_server(value: &Value) -> Result<(), SchemaError> {
    validate(&SCHEMAS.mcp_server, value)
}

pub fn validate_a2a_agent(value: &Value) -> Result<(), SchemaError> {
    validate(&SCHEMAS.a2a_agent, value)
}

/// Build a resource's canonical JSON Schema from its struct via `schemars`,
/// the single source of field shapes and per-field constraints.
///
/// `nullable_options` controls schemars' `Option<T>` representation: `false`
/// keeps optional fields plain-but-absent (`type: string`), matching the wire
/// shape of resources that never receive an explicit `null` (cp-api omits
/// unset fields); `true` keeps the default nullable form (`type: [string,
/// null]`) for resources whose schema deliberately accepts `null` (e.g.
/// ApiKey `team_id`/`user_id`).
///
/// Both the runtime validators in [`Schemas::compile`] and the `dump-schema`
/// binary that emits `schemas/resources/*.json` build from these producers, so
/// the published schema and the enforced schema are the same object by
/// construction — no hand-maintained second copy to drift.
fn struct_root_schema<T: schemars::JsonSchema>(nullable_options: bool) -> Value {
    use schemars::gen::{SchemaGenerator, SchemaSettings};

    let settings = SchemaSettings::draft07().with(|s| {
        s.option_add_null_type = nullable_options;
    });
    let root = SchemaGenerator::new(settings).into_root_schema_for::<T>();
    serde_json::to_value(root).expect("resource schema serializes to JSON")
}

/// Canonical JSON Schema for the `model` resource: the [`Model`] struct plus
/// the one cross-field invariant `schemars` cannot express
/// ([`super::model::model_one_of`] — the direct/routing/ensemble XOR).
///
/// [`Model`]: crate::models::Model
pub fn model_root_schema() -> Value {
    let mut schema = struct_root_schema::<crate::models::Model>(false);
    schema
        .as_object_mut()
        .expect("model root schema is a JSON object")
        .insert("oneOf".to_string(), super::model::model_one_of());
    schema
}

/// Canonical JSON Schema for the `api_key` resource, derived from the
/// [`ApiKey`](crate::models::ApiKey) struct. Uses the default nullable
/// `Option` representation so `team_id`/`user_id` keep accepting an explicit
/// `null` (cp-api sends `null` to clear team/owner), matching the resource's
/// wire contract.
pub fn apikey_root_schema() -> Value {
    struct_root_schema::<crate::models::ApiKey>(true)
}

/// Canonical JSON Schema for the `provider_key` resource, derived from the
/// [`ProviderKey`](crate::models::ProviderKey) struct. Uses the nullable
/// `Option` representation (`true`): `TelemetryTags` carries fields cp-api
/// sends as explicit `null` (`branded_provider`/`pk_label`/`byo_label`), and
/// keeping all optionals nullable matches the resource's wire contract.
pub fn provider_key_root_schema() -> Value {
    struct_root_schema::<crate::models::ProviderKey>(true)
}

/// Canonical JSON Schema for the `mcp_server` resource, derived from the
/// [`McpServer`](crate::models::McpServer) struct. Uses the nullable `Option`
/// representation (`true`) so the optional fields (`secret`, `client_id`,
/// `token_url`, `scopes`, `timeout_ms`) accept an explicit `null` as well as
/// being absent, matching the resource's wire contract. The `transport` /
/// `auth_type` closed sets come from the
/// [`McpTransport`](crate::models::McpTransport) /
/// [`McpAuthType`](crate::models::McpAuthType) enums. The per-`auth_type`
/// credential coupling is intentionally not encoded here (see the note on the
/// struct); the schema stays permissive and write paths enforce it.
pub fn mcp_server_root_schema() -> Value {
    let mut schema = struct_root_schema::<crate::models::McpServer>(true);
    if let Some(Value::Object(defs)) = schema.get_mut("definitions") {
        title_single_value_enum_variants(
            defs,
            "McpAuthType",
            &[
                ("none", "No authentication"),
                ("bearer", "Bearer token"),
                ("api_key", "API key"),
                ("oauth2", "OAuth 2.0 client credentials"),
            ],
        );
        title_single_value_enum_variants(
            defs,
            "McpTransport",
            &[("streamable_http", "Streamable HTTP")],
        );
    }
    schema
}

pub fn a2a_agent_root_schema() -> Value {
    let mut schema = struct_root_schema::<crate::models::A2aAgent>(true);
    if let Some(Value::Object(defs)) = schema.get_mut("definitions") {
        title_single_value_enum_variants(
            defs,
            "A2aAuthType",
            &[
                ("none", "No authentication"),
                ("bearer", "Bearer token"),
                ("api_key", "API key"),
            ],
        );
        title_single_value_enum_variants(
            defs,
            "A2aProtocolVersion",
            &[("1.0", "A2A 1.0"), ("0.3", "A2A 0.3")],
        );
    }
    schema
}

fn title_single_value_enum_variants(
    defs: &mut serde_json::Map<String, Value>,
    schema_name: &str,
    titles: &[(&str, &str)],
) {
    let Some(Value::Array(branches)) = defs.get_mut(schema_name).and_then(|d| d.get_mut("oneOf"))
    else {
        return;
    };
    for branch in branches.iter_mut() {
        let Some(branch) = branch.as_object_mut() else {
            continue;
        };
        let Some(value) = branch
            .get("enum")
            .and_then(|v| v.as_array())
            .and_then(|values| values.first())
            .and_then(|v| v.as_str())
        else {
            continue;
        };
        if let Some((_, title)) = titles.iter().find(|(expected, _)| *expected == value) {
            branch
                .entry("title".to_string())
                .or_insert_with(|| Value::String((*title).to_string()));
        }
    }
}

/// Canonical JSON Schema for the `guardrail` resource, derived from the
/// [`Guardrail`](crate::models::Guardrail) struct. `schemars` renders the
/// internally-tagged `GuardrailKind` as a native top-level `oneOf`; the
/// top-level object and its branches are intentionally open (matching the
/// hand-written schema — unknown inner fields are caught by serde at
/// deserialize). Three things need fixing up:
///
/// 1. The tagged sub-enums (`KeywordPattern`/`BedrockAWSCredentials`/
///    `BedrockLatencyMode`) lose `deny_unknown_fields` in their `oneOf`
///    branches, so each is re-closed with `additionalProperties: false`.
/// 2. The stringly-typed moderation fields carry closed sets the hand-written
///    schema enforced via `enum`. They stay `String` on the struct (their
///    values flow through `aisix-guardrails` as strings; converting them to
///    Rust enums would churn that crate's processing), so the closed set is
///    injected here into the relevant property.
/// 3. `schemars` leaves discriminator tag fields and collection item schemas
///    without descriptions, so the public schema fills those gaps.
/// 4. `created_at` republishes its `date-time` format (annotation-only).
pub fn guardrail_root_schema() -> Value {
    let mut schema = struct_root_schema::<crate::models::Guardrail>(false);
    let obj = schema
        .as_object_mut()
        .expect("guardrail root schema is a JSON object");

    if let Some(Value::Object(defs)) = obj.get_mut("definitions") {
        for name in [
            "KeywordPattern",
            "BedrockAWSCredentials",
            "BedrockLatencyMode",
        ] {
            if let Some(Value::Array(branches)) =
                defs.get_mut(name).and_then(|d| d.get_mut("oneOf"))
            {
                for branch in branches.iter_mut() {
                    if let Some(b) = branch.as_object_mut() {
                        b.insert("additionalProperties".to_string(), json!(false));
                    }
                }
            }
        }

        set_definition_property_enum(
            defs,
            "PiiDetectorConfig",
            "type",
            json!([
                "email",
                "china_mobile",
                "china_id_card",
                "bank_card",
                "us_ssn",
                "ip_address",
                "api_key",
                "jwt",
                "private_key"
            ]),
        );
        set_definition_property_enum(
            defs,
            "PiiDetectorConfig",
            "action",
            json!(["mask", "block"]),
        );
        set_definition_property_enum(defs, "PiiCustomPattern", "action", json!(["mask", "block"]));
        set_definition_property_enum(
            defs,
            "PresidioEntityConfig",
            "action",
            json!(["mask", "block"]),
        );
        set_definition_variant_property_description(
            defs,
            "BedrockAWSCredentials",
            "static",
            "kind",
            "Credential mode for explicitly configured AWS access keys.",
        );
        set_definition_variant_property_description(
            defs,
            "BedrockLatencyMode",
            "serial",
            "kind",
            "Latency mode that waits for the Bedrock guardrail response.",
        );
        set_definition_variant_property_description(
            defs,
            "BedrockLatencyMode",
            "timed",
            "kind",
            "Latency mode that stops waiting after `timeout_ms`.",
        );
        set_definition_variant_property_description(
            defs,
            "KeywordPattern",
            "literal",
            "kind",
            "Pattern type for matching the value as plain text.",
        );
        set_definition_variant_property_description(
            defs,
            "KeywordPattern",
            "regex",
            "kind",
            "Pattern type for matching the value as a regular expression.",
        );
        set_definition_variant_property_description(
            defs,
            "KeywordPattern",
            "literal",
            "value",
            "Literal string to match.",
        );
        set_definition_variant_property_description(
            defs,
            "KeywordPattern",
            "regex",
            "value",
            "Regular expression pattern to match.",
        );
    }

    if let Some(Value::Array(branches)) = obj.get_mut("oneOf") {
        for branch in branches.iter_mut() {
            let Some(b) = branch.as_object_mut() else {
                continue;
            };
            let Some(kind) = branch_kind(b).map(str::to_owned) else {
                continue;
            };
            if let Some(description) = guardrail_kind_description(&kind) {
                set_property_description(b, "kind", description);
            }
            match kind.as_str() {
                "azure_content_safety_text_moderation" => {
                    set_property_enum(
                        b,
                        "output_type",
                        json!(["FourSeverityLevels", "EightSeverityLevels"]),
                    );
                    set_property_enum(
                        b,
                        "text_source",
                        json!(["concatenate_user_content", "concatenate_all_content"]),
                    );
                    set_property_enum(
                        b,
                        "stream_processing_mode",
                        json!(["window", "buffer_full"]),
                    );
                    set_property_items_enum(
                        b,
                        "categories",
                        json!(["Hate", "Sexual", "SelfHarm", "Violence"]),
                    );
                    set_property_items_description(
                        b,
                        "categories",
                        "Azure content category to analyze.",
                    );
                    set_property_items_description(
                        b,
                        "blocklist_names",
                        "Azure blocklist name to match against.",
                    );
                    set_property_additional_properties_description(
                        b,
                        "severity_threshold_by_category",
                        "Severity threshold for the category key.",
                    );
                }
                "aliyun_text_moderation" => {
                    set_property_enum(b, "risk_level_threshold", json!(["low", "medium", "high"]));
                    set_property_enum(
                        b,
                        "stream_processing_mode",
                        json!(["window", "buffer_full"]),
                    );
                }
                "pii" => {
                    set_property_enum(b, "default_action", json!(["mask", "block"]));
                }
                "openai_moderation" => {
                    set_property_additional_properties_description(
                        b,
                        "category_thresholds",
                        "Score threshold for the category key.",
                    );
                }
                "presidio" => {
                    set_property_enum(b, "default_action", json!(["mask", "block"]));
                    set_property_enum(b, "operator", json!(["replace", "mask", "hash", "redact"]));
                }
                _ => {}
            }
        }
    }

    if let Some(created_at) = obj
        .get_mut("properties")
        .and_then(|p| p.get_mut("created_at"))
        .and_then(Value::as_object_mut)
    {
        created_at.insert("format".to_string(), json!("date-time"));
    }

    schema
}

/// Set a closed `enum` on a oneOf branch's property (for stringly-typed fields
/// whose closed set lives only in the schema, not the Rust type).
fn set_property_enum(branch: &mut serde_json::Map<String, Value>, field: &str, values: Value) {
    if let Some(Value::Object(properties)) = branch.get_mut("properties") {
        set_enum(properties, field, values);
    }
}

fn set_definition_property_enum(
    defs: &mut serde_json::Map<String, Value>,
    definition: &str,
    field: &str,
    values: Value,
) {
    if let Some(Value::Object(properties)) = defs
        .get_mut(definition)
        .and_then(|d| d.get_mut("properties"))
    {
        set_enum(properties, field, values);
    }
}

fn set_definition_variant_property_description(
    defs: &mut serde_json::Map<String, Value>,
    definition: &str,
    variant_kind: &str,
    field: &str,
    description: &str,
) {
    let Some(Value::Array(branches)) = defs.get_mut(definition).and_then(|d| d.get_mut("oneOf"))
    else {
        return;
    };
    for branch in branches {
        let Some(branch) = branch.as_object_mut() else {
            continue;
        };
        if branch_kind(branch) == Some(variant_kind) {
            set_property_description(branch, field, description);
        }
    }
}

fn guardrail_kind_description(kind: &str) -> Option<&'static str> {
    match kind {
        "keyword" => Some("Guardrail provider type for literal and regular expression matching."),
        "bedrock" => Some("Guardrail provider type for Amazon Bedrock Guardrails."),
        "azure_content_safety" => Some("Guardrail provider type for Azure Prompt Shield."),
        "azure_content_safety_text_moderation" => {
            Some("Guardrail provider type for Azure text moderation.")
        }
        "aliyun_text_moderation" => Some("Guardrail provider type for Aliyun text moderation."),
        "pii" => {
            Some("Guardrail provider type for in-process sensitive-data detection and redaction.")
        }
        "lakera" => Some("Guardrail provider type for Lakera Guard screening."),
        "openai_moderation" => Some("Guardrail provider type for the OpenAI Moderation API."),
        "presidio" => {
            Some("Guardrail provider type for self-hosted Microsoft Presidio PII detection and anonymization.")
        }
        _ => None,
    }
}

fn set_enum(properties: &mut serde_json::Map<String, Value>, field: &str, values: Value) {
    if let Some(prop) = properties.get_mut(field).and_then(Value::as_object_mut) {
        prop.insert("enum".to_string(), values);
    }
}

fn set_property_description(
    branch: &mut serde_json::Map<String, Value>,
    field: &str,
    description: &str,
) {
    if let Some(prop) = branch
        .get_mut("properties")
        .and_then(|p| p.get_mut(field))
        .and_then(Value::as_object_mut)
    {
        prop.entry("description".to_string())
            .or_insert_with(|| Value::String(description.to_string()));
    }
}

/// Like [`set_property_enum`] but for the `items` of an array property.
fn set_property_items_enum(
    branch: &mut serde_json::Map<String, Value>,
    field: &str,
    values: Value,
) {
    if let Some(items) = branch
        .get_mut("properties")
        .and_then(|p| p.get_mut(field))
        .and_then(|f| f.get_mut("items"))
        .and_then(Value::as_object_mut)
    {
        items.insert("enum".to_string(), values);
    }
}

fn set_property_items_description(
    branch: &mut serde_json::Map<String, Value>,
    field: &str,
    description: &str,
) {
    if let Some(items) = branch
        .get_mut("properties")
        .and_then(|p| p.get_mut(field))
        .and_then(|f| f.get_mut("items"))
        .and_then(Value::as_object_mut)
    {
        items
            .entry("description".to_string())
            .or_insert_with(|| Value::String(description.to_string()));
    }
}

fn set_property_additional_properties_description(
    branch: &mut serde_json::Map<String, Value>,
    field: &str,
    description: &str,
) {
    if let Some(additional_properties) = branch
        .get_mut("properties")
        .and_then(|p| p.get_mut(field))
        .and_then(|f| f.get_mut("additionalProperties"))
        .and_then(Value::as_object_mut)
    {
        additional_properties
            .entry("description".to_string())
            .or_insert_with(|| Value::String(description.to_string()));
    }
}

/// Canonical JSON Schema for the `cache_policy` resource, derived from the
/// [`CachePolicy`](crate::models::CachePolicy) struct. The struct intentionally
/// has no `deny_unknown_fields`, so the schema omits `additionalProperties`
/// (i.e. `true`) — forward-compat fields from a newer cp-api are tolerated.
pub fn cache_policy_root_schema() -> Value {
    struct_root_schema::<crate::models::CachePolicy>(false)
}

/// Canonical JSON Schema for the `observability_exporter` resource, derived
/// from the [`ObservabilityExporter`](crate::models::ObservabilityExporter)
/// struct. `schemars` renders the internally-tagged `ExporterKind` as a native
/// top-level `oneOf`, but two things need fixing up by hand:
///
/// 1. `schemars` drops `deny_unknown_fields` inside tagged-enum branches, and
///    serde does not enforce it there either, so each branch is re-closed with
///    `additionalProperties: false` (rejecting a smuggled plaintext secret).
///    Because a closed branch only lists its own kind's fields, the shared
///    top-level `name`/`enabled` are copied into every branch.
/// 2. The `object_store` cloud-identity cross-field rule (cloud_identity ⇒
///    provider ∈ {s3,gcs} and no credential_ref; otherwise credential_ref
///    required) is injected as an `allOf`/`if`/`then`/`else` — `schemars` can't
///    derive cross-field constraints.
///
/// Re-closing each branch also rejects cross-kind field leakage (e.g. a
/// `datadog` exporter carrying an otlp `project`) that the previous
/// single-union-object validator silently accepted; no valid config mixes kinds.
pub fn observability_exporter_root_schema() -> Value {
    let mut schema = struct_root_schema::<crate::models::ObservabilityExporter>(false);
    let obj = schema
        .as_object_mut()
        .expect("observability_exporter root schema is a JSON object");

    let top_props = obj
        .get("properties")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();

    if let Some(Value::Array(branches)) = obj.get_mut("oneOf") {
        for branch in branches.iter_mut() {
            let Some(branch_obj) = branch.as_object_mut() else {
                continue;
            };
            let is_object_store = branch_kind(branch_obj) == Some("object_store");

            let props = branch_obj
                .entry("properties".to_string())
                .or_insert_with(|| json!({}));
            if let Some(props_obj) = props.as_object_mut() {
                for key in ["name", "enabled"] {
                    if let Some(v) = top_props.get(key) {
                        props_obj
                            .entry(key.to_string())
                            .or_insert_with(|| v.clone());
                    }
                }
            }

            if is_object_store {
                branch_obj.insert(
                    "allOf".to_string(),
                    json!([{
                        "if": {
                            "required": ["auth_mode"],
                            "properties": { "auth_mode": { "const": "cloud_identity" } }
                        },
                        "then": { "properties": { "provider": { "enum": ["s3", "gcs"] } } },
                        "else": { "required": ["credential_ref"] }
                    }]),
                );
            }

            branch_obj.insert("additionalProperties".to_string(), json!(false));
        }
    }
    schema
}

/// The `kind` discriminator value of a schemars-generated tagged-enum `oneOf`
/// branch, whether rendered as a `const` or a single-element `enum`.
fn branch_kind(branch: &serde_json::Map<String, Value>) -> Option<&str> {
    let kind = branch.get("properties")?.get("kind")?;
    if let Some(c) = kind.get("const").and_then(Value::as_str) {
        return Some(c);
    }
    kind.get("enum")?.as_array()?.first()?.as_str()
}

/// Canonical JSON Schema for the `rate_limit_policy` resource, derived from the
/// [`RateLimitPolicy`](crate::models::RateLimitPolicy) struct (the `scope`/
/// `window` closed sets come from the `PolicyScope`/`PolicyWindow` enums) plus
/// the one cross-field invariant `schemars` can't express: at least one of
/// `max_requests`/`max_tokens` must be set
/// ([`super::rate_limit_policy::rate_limit_policy_any_of`]).
pub fn rate_limit_policy_root_schema() -> Value {
    let mut schema = struct_root_schema::<crate::models::RateLimitPolicy>(false);
    schema
        .as_object_mut()
        .expect("rate_limit_policy root schema is a JSON object")
        .insert(
            "anyOf".to_string(),
            super::rate_limit_policy::rate_limit_policy_any_of(),
        );
    schema
}

/// Canonical JSON Schema for the `guardrail_attachment` resource, derived from
/// the [`GuardrailAttachment`](crate::models::GuardrailAttachment) struct. Uses
/// the nullable `Option` representation (`scope_id` is `null` for `env`-scoped
/// attachments) and stays open (no `deny_unknown_fields`): cp-api includes an
/// `env_id` the DP ignores.
pub fn guardrail_attachment_root_schema() -> Value {
    struct_root_schema::<crate::models::GuardrailAttachment>(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn model_happy_path_passes() {
        let v = json!({
            "display_name": "my-gpt4",
            "provider": "openai",
            "model_name": "gpt-4o",
            "provider_key_id": "11111111-1111-1111-1111-111111111111",
            "timeout": 30000,
            "rate_limit": {"rpm": 100}
        });
        validate_model(&v).unwrap();
    }

    #[test]
    fn model_routing_form_passes() {
        let v = json!({
            "display_name": "router-1",
            "routing": {
                "strategy": "round_robin",
                "targets": [{"model": "my-gpt4"}, {"model": "my-claude"}]
            }
        });
        validate_model(&v).unwrap();
    }

    #[test]
    fn model_ensemble_form_passes() {
        let v = json!({
            "display_name": "council",
            "ensemble": {
                "panel": [
                    {"model": "my-gpt4", "temperature": 0.5},
                    {"model": "my-claude", "temperature": 1.0}
                ],
                "judge": {"model": "my-opus"},
                "min_responses": 2,
                "timeout_ms": 45000
            }
        });
        validate_model(&v).unwrap();
    }

    #[test]
    fn model_ensemble_can_be_ip_restricted_and_rate_limited() {
        // Top-level gates apply to the ensemble entry model too.
        let v = json!({
            "display_name": "council",
            "ensemble": {
                "panel": [{"model": "a"}, {"model": "b"}],
                "judge": {"model": "j"}
            },
            "allowed_cidrs": ["10.0.0.0/8"],
            "rate_limit": {"rpm": 60}
        });
        validate_model(&v).unwrap();
    }

    #[test]
    fn model_rate_limit_accepts_all_request_windows_incl_rps_rph() {
        // Regression for #644: the inline rate_limit schema is derived from the
        // RateLimit struct (#638), so every request-count window — rps/rpm/rph/
        // rpd — alongside the token windows and concurrency must be accepted,
        // not just rpm/rpd/tpm/tpd/concurrency.
        let v = json!({
            "display_name": "my-gpt4",
            "provider": "openai",
            "model_name": "gpt-4o",
            "provider_key_id": "11111111-1111-1111-1111-111111111111",
            "rate_limit": {
                "rps": 10, "rpm": 100, "rph": 1000, "rpd": 10000,
                "tpm": 100000, "tpd": 1000000, "concurrency": 5
            }
        });
        validate_model(&v).unwrap();
    }

    #[test]
    fn model_ensemble_with_direct_fields_fails() {
        // ensemble is mutually exclusive with the direct upstream triple.
        let v = json!({
            "display_name": "x",
            "provider": "openai",
            "model_name": "gpt-4o",
            "provider_key_id": "pk-1",
            "ensemble": {
                "panel": [{"model": "a"}],
                "judge": {"model": "j"}
            }
        });
        assert!(validate_model(&v).is_err());
    }

    #[test]
    fn model_ensemble_with_routing_fails() {
        // A model can't be both an ensemble and a router.
        let v = json!({
            "display_name": "x",
            "routing": {"targets": [{"model": "a"}]},
            "ensemble": {
                "panel": [{"model": "a"}],
                "judge": {"model": "j"}
            }
        });
        assert!(validate_model(&v).is_err());
    }

    #[test]
    fn model_ensemble_missing_judge_fails() {
        let v = json!({
            "display_name": "x",
            "ensemble": {
                "panel": [{"model": "a"}, {"model": "b"}]
            }
        });
        assert!(validate_model(&v).is_err());
    }

    #[test]
    fn model_ensemble_empty_panel_fails() {
        let v = json!({
            "display_name": "x",
            "ensemble": {
                "panel": [],
                "judge": {"model": "j"}
            }
        });
        assert!(validate_model(&v).is_err());
    }

    #[test]
    fn model_ensemble_unknown_panel_field_fails() {
        let v = json!({
            "display_name": "x",
            "ensemble": {
                "panel": [{"model": "a", "bogus": true}],
                "judge": {"model": "j"}
            }
        });
        assert!(validate_model(&v).is_err());
    }

    // ---- semantic-routing + embedding-modality schema tests (#641) ----

    #[test]
    fn model_semantic_form_passes() {
        let v = json!({
            "display_name": "prod-chat",
            "semantic": {
                "embedding_model": "bge-m3",
                "routes": [
                    {
                        "name": "legal",
                        "target": "claude-opus",
                        "description": "Contract & legal risk analysis",
                        "examples": ["分析这份合同里的潜在风险", "Review this NDA"],
                        "threshold": 0.8
                    },
                    {"name": "translate", "target": "gpt-4o-mini", "examples": ["帮我翻译这句话"]}
                ],
                "default": "gpt-4o",
                "match": {"distance_metric": "cosine", "aggregation": "max", "threshold": 0.75},
                "embedding_timeout_ms": 500,
                "on_embedding_failure": {"target": "gpt-4o-mini"}
            }
        });
        validate_model(&v).unwrap();
    }

    #[test]
    fn model_semantic_minimal_form_passes() {
        let v = json!({
            "display_name": "prod-chat",
            "semantic": {
                "embedding_model": "bge-m3",
                "routes": [{"name": "a", "target": "m", "examples": ["hi"]}],
                "default": "gpt-4o",
                "match": {"threshold": 0.5}
            }
        });
        validate_model(&v).unwrap();
    }

    #[test]
    fn model_semantic_can_be_ip_restricted_and_rate_limited() {
        // Top-level gates apply to the semantic router entry too.
        let v = json!({
            "display_name": "prod-chat",
            "semantic": {
                "embedding_model": "e",
                "routes": [{"name": "a", "target": "m", "examples": ["x"]}],
                "default": "d",
                "match": {"threshold": 0.5}
            },
            "allowed_cidrs": ["10.0.0.0/8"],
            "rate_limit": {"rpm": 60}
        });
        validate_model(&v).unwrap();
    }

    #[test]
    fn model_semantic_on_embedding_failure_accepts_bare_modes() {
        for mode in ["default", "fail"] {
            let v = json!({
                "display_name": "prod-chat",
                "semantic": {
                    "embedding_model": "e",
                    "routes": [{"name": "a", "target": "m", "examples": ["x"]}],
                    "default": "d",
                    "match": {"threshold": 0.5},
                    "on_embedding_failure": mode
                }
            });
            validate_model(&v).unwrap_or_else(|e| panic!("mode {mode:?} must validate: {e:?}"));
        }
    }

    #[test]
    fn model_semantic_with_direct_fields_fails() {
        // semantic is mutually exclusive with the direct upstream triple.
        let v = json!({
            "display_name": "x",
            "provider": "openai",
            "model_name": "gpt-4o",
            "provider_key_id": "pk-1",
            "semantic": {
                "embedding_model": "e",
                "routes": [{"name": "a", "target": "m", "examples": ["x"]}],
                "default": "d",
                "match": {"threshold": 0.5}
            }
        });
        assert!(validate_model(&v).is_err());
    }

    #[test]
    fn model_semantic_with_routing_fails() {
        let v = json!({
            "display_name": "x",
            "routing": {"targets": [{"model": "a"}]},
            "semantic": {
                "embedding_model": "e",
                "routes": [{"name": "a", "target": "m", "examples": ["x"]}],
                "default": "d",
                "match": {"threshold": 0.5}
            }
        });
        assert!(validate_model(&v).is_err());
    }

    #[test]
    fn model_semantic_missing_required_fields_fails() {
        // Missing `default`.
        let v = json!({
            "display_name": "x",
            "semantic": {
                "embedding_model": "e",
                "routes": [{"name": "a", "target": "m", "examples": ["x"]}],
                "match": {"threshold": 0.5}
            }
        });
        assert!(validate_model(&v).is_err());
    }

    #[test]
    fn model_semantic_empty_routes_fails() {
        let v = json!({
            "display_name": "x",
            "semantic": {
                "embedding_model": "e",
                "routes": [],
                "default": "d",
                "match": {"threshold": 0.5}
            }
        });
        assert!(validate_model(&v).is_err());
    }

    #[test]
    fn model_semantic_route_without_examples_fails() {
        // examples-only matching: a route needs at least one example.
        let v = json!({
            "display_name": "x",
            "semantic": {
                "embedding_model": "e",
                "routes": [{"name": "a", "target": "m", "examples": []}],
                "default": "d",
                "match": {"threshold": 0.5}
            }
        });
        assert!(validate_model(&v).is_err());
    }

    #[test]
    fn model_semantic_threshold_out_of_range_fails() {
        let v = json!({
            "display_name": "x",
            "semantic": {
                "embedding_model": "e",
                "routes": [{"name": "a", "target": "m", "examples": ["x"]}],
                "default": "d",
                "match": {"threshold": 1.5}
            }
        });
        assert!(validate_model(&v).is_err());
    }

    #[test]
    fn model_embedding_modality_on_direct_passes() {
        // An embedding model is a direct model that also carries the
        // embedding-modality block.
        let v = json!({
            "display_name": "bge-m3",
            "provider": "openai",
            "model_name": "bge-m3",
            "provider_key_id": "pk-1",
            "embedding": {"dimensions": 1024, "normalize": false}
        });
        validate_model(&v).unwrap();
    }

    #[test]
    fn model_embedding_without_dimensions_fails() {
        let v = json!({
            "display_name": "bge-m3",
            "provider": "openai",
            "model_name": "bge-m3",
            "provider_key_id": "pk-1",
            "embedding": {"normalize": true}
        });
        assert!(validate_model(&v).is_err());
    }

    #[test]
    fn model_embedding_on_routing_fails() {
        // The embedding block is modality metadata on a direct model — it
        // has no meaning on a virtual router.
        let v = json!({
            "display_name": "x",
            "routing": {"targets": [{"model": "a"}]},
            "embedding": {"dimensions": 1024}
        });
        assert!(validate_model(&v).is_err());
    }

    #[test]
    fn model_embedding_on_semantic_fails() {
        let v = json!({
            "display_name": "x",
            "semantic": {
                "embedding_model": "e",
                "routes": [{"name": "a", "target": "m", "examples": ["x"]}],
                "default": "d",
                "match": {"threshold": 0.5}
            },
            "embedding": {"dimensions": 1024}
        });
        assert!(validate_model(&v).is_err());
    }

    #[test]
    fn model_allowed_cidrs_passes_on_direct_and_routing() {
        // Direct model with an IP allowlist (#557).
        let direct = json!({
            "display_name": "ip-restricted",
            "provider": "openai",
            "model_name": "gpt-4o",
            "provider_key_id": "11111111-1111-1111-1111-111111111111",
            "allowed_cidrs": ["10.0.0.0/8", "2001:db8::/32"]
        });
        validate_model(&direct).unwrap();

        // Routing (Model Group) model can also be IP-restricted — the gate
        // binds to the requested model name regardless of its shape.
        let routing = json!({
            "display_name": "router-restricted",
            "routing": {
                "strategy": "failover",
                "targets": [{"model": "my-gpt4"}]
            },
            "allowed_cidrs": ["10.0.0.0/8"]
        });
        validate_model(&routing).unwrap();
    }

    #[test]
    fn model_missing_display_name_fails() {
        let v = json!({
            "provider": "openai",
            "model_name": "gpt-4o",
            "provider_key_id": "pk-1"
        });
        let err = validate_model(&v).unwrap_err();
        assert!(err.message.to_lowercase().contains("display_name"));
    }

    /// Closed-enum on `provider` was the cause of api7/AISIX-Cloud#417
    /// — any catalog vendor not in the DP enum (`xai`, `openrouter`,
    /// future long-tail) failed schema validation at snapshot load
    /// and silently disappeared from dispatch. Phase A opened the
    /// field to a free-form string; the only invariant left is
    /// `minLength: 1`.
    #[test]
    fn model_accepts_arbitrary_provider_string() {
        // Every real models.dev catalog id must pass. `wafer.ai` is
        // the load-bearing example: one real vendor has a dot in its
        // id, so the schema pattern must accept `.` — rejecting it
        // would re-create the #417 bug class for that vendor.
        // `fireworks-ai` is the canonical hyphenated example.
        for provider in [
            "openai",
            "xai",
            "openrouter",
            "wafer.ai",
            "fireworks-ai",
            "togetherai",
            "this-is-some-new-vendor",
        ] {
            let v = json!({
                "display_name": "x",
                "provider": provider,
                "model_name": "x",
                "provider_key_id": "pk-1"
            });
            validate_model(&v).unwrap_or_else(|err| {
                panic!("provider {provider:?} should validate after #302 Phase A; got {err:?}")
            });
        }
    }

    /// Pattern guards against log-injection / cardinality explosion.
    /// Each rejected case here is a string the round-1 audit listed
    /// as a concern.
    #[test]
    fn model_rejects_provider_strings_outside_pattern() {
        for bad in [
            "\nfake_log_line",
            "openai\nline2",
            "with space",
            "UPPER",
            ".leading-dot",
            "-leading-hyphen",
            "_leading-underscore",
            "trailing-byte\0",
        ] {
            let v = json!({
                "display_name": "x",
                "provider": bad,
                "model_name": "x",
                "provider_key_id": "pk-1"
            });
            assert!(
                validate_model(&v).is_err(),
                "provider {bad:?} MUST be rejected by the pattern guard",
            );
        }
    }

    /// `maxLength: 64` bounds Prometheus label cardinality. The
    /// longest real models.dev catalog id today is ~19 chars; the
    /// cap is generous but finite. A regression that drops the cap
    /// would let a crafted ~10KB vendor string flow into metric
    /// labels.
    #[test]
    fn model_rejects_provider_string_over_maxlength() {
        let too_long = "a".repeat(65);
        let v = json!({
            "display_name": "x",
            "provider": too_long,
            "model_name": "x",
            "provider_key_id": "pk-1"
        });
        assert!(
            validate_model(&v).is_err(),
            "provider string > 64 chars MUST be rejected (Prometheus cardinality guard)",
        );
    }

    #[test]
    fn model_rejects_empty_provider_string() {
        let v = json!({
            "display_name": "x",
            "provider": "",
            "model_name": "x",
            "provider_key_id": "pk-1"
        });
        assert!(
            validate_model(&v).is_err(),
            "empty `provider` must fail (minLength: 1)"
        );
    }

    #[test]
    fn model_direct_with_routing_block_fails() {
        // Direct + routing both present violates the oneOf XOR.
        let v = json!({
            "display_name": "x",
            "provider": "openai",
            "model_name": "gpt-4o",
            "provider_key_id": "pk-1",
            "routing": {"targets": [{"model": "y"}]}
        });
        assert!(validate_model(&v).is_err());
    }

    #[test]
    fn model_routing_with_provider_key_id_fails() {
        // Router can't carry provider_key_id — that lives on the
        // target Models the router fans out to.
        let v = json!({
            "display_name": "router-1",
            "provider_key_id": "pk-1",
            "routing": {"targets": [{"model": "y"}]}
        });
        assert!(validate_model(&v).is_err());
    }

    #[test]
    fn model_direct_missing_provider_key_id_fails() {
        // Direct model needs all three of provider / model_name /
        // provider_key_id.
        let v = json!({
            "display_name": "x",
            "provider": "openai",
            "model_name": "gpt-4o"
        });
        assert!(validate_model(&v).is_err());
    }

    #[test]
    fn model_rejects_additional_top_level() {
        let v = json!({
            "display_name":"x","provider":"openai","model_name":"g","provider_key_id":"pk-1",
            "rogue": 1
        });
        assert!(validate_model(&v).is_err());
    }

    #[test]
    fn apikey_happy_path_passes() {
        let v = json!({"key_hash":"9df37f5e7cbc3c391d872742b5f286c242e733a09add9eeaa4d26a599bd90b20","allowed_models":["a","b"]});
        validate_apikey(&v).unwrap();
    }

    #[test]
    fn apikey_missing_allowed_models_fails() {
        let v =
            json!({"key_hash":"9df37f5e7cbc3c391d872742b5f286c242e733a09add9eeaa4d26a599bd90b20"});
        let err = validate_apikey(&v).unwrap_err();
        assert!(err.message.to_lowercase().contains("allowed_models"));
    }

    #[test]
    fn apikey_empty_allowed_models_is_valid_but_denies_all() {
        // Schema permits []; runtime ApiKey::can_access enforces deny-all.
        let v = json!({"key_hash":"9df37f5e7cbc3c391d872742b5f286c242e733a09add9eeaa4d26a599bd90b20","allowed_models":[]});
        validate_apikey(&v).unwrap();
    }

    #[test]
    fn apikey_with_team_and_user_fields_passes() {
        let v = json!({
            "key_hash":"9df37f5e7cbc3c391d872742b5f286c242e733a09add9eeaa4d26a599bd90b20",
            "allowed_models":["gpt-4o"],
            "team_id": "team-uuid-1",
            "user_id": "member-uuid-1"
        });
        validate_apikey(&v).unwrap();
    }

    #[test]
    fn apikey_with_null_team_and_user_fields_passes() {
        let v = json!({
            "key_hash":"9df37f5e7cbc3c391d872742b5f286c242e733a09add9eeaa4d26a599bd90b20",
            "allowed_models":["gpt-4o"],
            "team_id": null,
            "user_id": null
        });
        validate_apikey(&v).unwrap();
    }

    #[test]
    fn apikey_rate_limit_accepts_rps_and_rph() {
        // Regression for #644: inline rate_limit on a caller API key must accept
        // the per-second and per-hour request windows too, not only
        // rpm/rpd/tpm/tpd/concurrency.
        let v = json!({
            "key_hash":"9df37f5e7cbc3c391d872742b5f286c242e733a09add9eeaa4d26a599bd90b20",
            "allowed_models":["gpt-4o"],
            "rate_limit": {"rps": 5, "rph": 500}
        });
        validate_apikey(&v).unwrap();
    }

    #[test]
    fn apikey_unknown_field_rejected() {
        let v = json!({
            "key_hash":"9df37f5e7cbc3c391d872742b5f286c242e733a09add9eeaa4d26a599bd90b20",
            "allowed_models":["a"],
            "bogus_field": true
        });
        assert!(validate_apikey(&v).is_err());
    }

    #[test]
    fn rate_limit_negative_value_rejected() {
        let v = json!({
            "display_name":"x","provider":"openai","model_name":"g","provider_key_id":"pk-1",
            "rate_limit": {"rpm": -1}
        });
        assert!(validate_model(&v).is_err());
    }

    #[test]
    fn direct_model_background_check_passes() {
        let v = json!({
            "display_name": "x",
            "provider": "openai",
            "model_name": "g",
            "provider_key_id": "pk-1",
            "background_model_check": {
                "enabled": true,
                "interval_seconds": 30,
                "timeout_seconds": 10,
                "prompt": "Respond with OK",
                "max_tokens": 8,
                "ignore_statuses": [408, 429],
                "stale_after_seconds": 90
            }
        });
        validate_model(&v).unwrap();
    }

    #[test]
    fn routing_model_background_check_fails() {
        let v = json!({
            "display_name": "router-1",
            "routing": {
                "targets": [{"model": "my-gpt4"}]
            },
            "background_model_check": {
                "enabled": true,
                "interval_seconds": 30,
                "timeout_seconds": 10,
                "prompt": "Respond with OK",
                "max_tokens": 8,
                "stale_after_seconds": 90
            }
        });
        assert!(validate_model(&v).is_err());
    }

    #[test]
    fn direct_model_cooldown_block_passes() {
        let v = json!({
            "display_name": "x",
            "provider": "openai",
            "model_name": "g",
            "provider_key_id": "pk-1",
            "cooldown": {
                "enabled": true,
                "default_seconds": 30,
                "max_seconds": 600,
                "honor_retry_after": true,
                "trigger_statuses": [401, 408, 429, 500, 502, 503, 504],
                "trigger_on_timeout": true,
                "trigger_on_transport": true
            }
        });
        validate_model(&v).unwrap();
    }

    #[test]
    fn cooldown_block_partial_override_passes() {
        // Only set one field — defaults fill the rest at runtime.
        let v = json!({
            "display_name": "x",
            "provider": "openai",
            "model_name": "g",
            "provider_key_id": "pk-1",
            "cooldown": {
                "default_seconds": 90
            }
        });
        validate_model(&v).unwrap();
    }

    #[test]
    fn routing_model_cooldown_block_fails() {
        // Cooldown is direct-model-only — routing models project to
        // their underlying targets and have no upstream of their own.
        let v = json!({
            "display_name": "router-1",
            "routing": { "targets": [{"model": "x"}] },
            "cooldown": { "default_seconds": 30 }
        });
        assert!(validate_model(&v).is_err());
    }

    #[test]
    fn routing_fallback_on_statuses_range_is_enforced() {
        // AISIX-Cloud#1012: entries outside 400-599 are rejected by the
        // same committed-schema validation the admin API and etcd watch
        // paths share; an in-range list passes.
        let bad = json!({
            "display_name": "router-fos",
            "routing": {
                "targets": [{"model": "x"}],
                "fallback_on_statuses": [300]
            }
        });
        assert!(validate_model(&bad).is_err());
        let good = json!({
            "display_name": "router-fos",
            "routing": {
                "targets": [{"model": "x"}],
                "fallback_on_statuses": [408, 422]
            }
        });
        validate_model(&good).unwrap();
    }

    #[test]
    fn cooldown_rejects_invalid_status_code() {
        let v = json!({
            "display_name": "x",
            "provider": "openai",
            "model_name": "g",
            "provider_key_id": "pk-1",
            "cooldown": { "trigger_statuses": [99] }
        });
        assert!(validate_model(&v).is_err());
    }

    #[test]
    fn cooldown_max_seconds_must_be_positive() {
        let v = json!({
            "display_name": "x",
            "provider": "openai",
            "model_name": "g",
            "provider_key_id": "pk-1",
            "cooldown": { "max_seconds": 0 }
        });
        assert!(validate_model(&v).is_err());
    }

    #[test]
    fn routing_when_all_unavailable_fail_passes() {
        let v = json!({
            "display_name": "router-1",
            "routing": {
                "targets": [{"model": "a"}],
                "when_all_unavailable": "fail"
            }
        });
        validate_model(&v).unwrap();
    }

    #[test]
    fn routing_when_all_unavailable_try_anyway_passes() {
        let v = json!({
            "display_name": "router-1",
            "routing": {
                "targets": [{"model": "a"}],
                "when_all_unavailable": "try_anyway"
            }
        });
        validate_model(&v).unwrap();
    }

    #[test]
    fn routing_when_all_unavailable_rejects_unknown_value() {
        let v = json!({
            "display_name": "router-1",
            "routing": {
                "targets": [{"model": "a"}],
                "when_all_unavailable": "yolo"
            }
        });
        assert!(validate_model(&v).is_err());
    }

    #[test]
    fn background_check_interval_below_min_fails() {
        // Minimum interval is 5s — guards misconfiguration from
        // burning provider quota on a 1s loop.
        let v = json!({
            "display_name": "x",
            "provider": "openai",
            "model_name": "g",
            "provider_key_id": "pk-1",
            "background_model_check": {
                "enabled": true,
                "interval_seconds": 1,
                "timeout_seconds": 10,
                "prompt": "Respond with OK",
                "max_tokens": 8,
                "stale_after_seconds": 90
            }
        });
        assert!(validate_model(&v).is_err());
    }

    #[test]
    fn background_check_rejects_invalid_ignore_status() {
        let v = json!({
            "display_name": "x",
            "provider": "openai",
            "model_name": "g",
            "provider_key_id": "pk-1",
            "background_model_check": {
                "enabled": true,
                "interval_seconds": 30,
                "timeout_seconds": 10,
                "prompt": "Respond with OK",
                "max_tokens": 8,
                "ignore_statuses": [99],
                "stale_after_seconds": 90
            }
        });
        assert!(validate_model(&v).is_err());
    }

    #[test]
    fn schemas_initialise_once() {
        let a = Arc::as_ptr(&*SCHEMAS);
        let b = Arc::as_ptr(&*SCHEMAS);
        assert_eq!(a, b);
    }

    #[test]
    fn guardrail_bedrock_serial_passes() {
        let v = json!({
            "name": "block-pii",
            "kind": "bedrock",
            "guardrail_id": "abcdefgh1234",
            "guardrail_version": "DRAFT",
            "region": "us-east-1",
            "aws_credentials": {
                "kind": "static",
                "access_key_id": "AKIAEXAMPLE",
                "secret_access_key": "PLAINTEXT"
            },
            "latency_mode": { "kind": "serial" }
        });
        validate_guardrail(&v).unwrap();
    }

    #[test]
    fn guardrail_bedrock_timed_with_valid_timeout_passes() {
        let v = json!({
            "name": "block-pii",
            "kind": "bedrock",
            "guardrail_id": "id",
            "guardrail_version": "1",
            "region": "us-east-1",
            "aws_credentials": {
                "kind": "static",
                "access_key_id": "AKIA",
                "secret_access_key": "S"
            },
            "latency_mode": { "kind": "timed", "timeout_ms": 500 }
        });
        validate_guardrail(&v).unwrap();
    }

    #[test]
    fn guardrail_bedrock_timeout_below_min_rejected() {
        let v = json!({
            "name": "g",
            "kind": "bedrock",
            "guardrail_id": "id",
            "guardrail_version": "1",
            "region": "us-east-1",
            "aws_credentials": { "kind": "static", "access_key_id": "AKIA" },
            "latency_mode": { "kind": "timed", "timeout_ms": 50 }
        });
        assert!(validate_guardrail(&v).is_err());
    }

    #[test]
    fn guardrail_bedrock_unknown_credential_kind_rejected() {
        let v = json!({
            "name": "g",
            "kind": "bedrock",
            "guardrail_id": "id",
            "guardrail_version": "1",
            "region": "us-east-1",
            "aws_credentials": { "kind": "role_arn", "access_key_id": "AKIA" },
            "latency_mode": { "kind": "serial" }
        });
        // Phase 4 will add role_arn; today it's rejected.
        assert!(validate_guardrail(&v).is_err());
    }

    #[test]
    fn guardrail_azure_content_safety_passes() {
        // Regression for #437: the loader JSON schema must accept the
        // azure_content_safety kind, not just the Rust struct. timeout_ms
        // omitted here — it's optional (defaults to 5000 on the struct).
        let v = json!({
            "name": "prompt-shield",
            "kind": "azure_content_safety",
            "hook_point": "input",
            "endpoint": "https://my-resource.cognitiveservices.azure.com",
            "api_key": "plaintext-key"
        });
        validate_guardrail(&v).unwrap();
    }

    #[test]
    fn guardrail_azure_content_safety_with_timeout_passes() {
        let v = json!({
            "name": "prompt-shield",
            "kind": "azure_content_safety",
            "endpoint": "https://r.cognitiveservices.azure.com",
            "api_key": "k",
            "timeout_ms": 3000
        });
        validate_guardrail(&v).unwrap();
    }

    #[test]
    fn guardrail_azure_content_safety_missing_api_key_rejected() {
        let v = json!({
            "name": "g",
            "kind": "azure_content_safety",
            "endpoint": "https://r.cognitiveservices.azure.com"
        });
        assert!(validate_guardrail(&v).is_err());
    }

    #[test]
    fn guardrail_azure_content_safety_max_timeout_passes() {
        // Guards the exact regression class of #437: the loader schema
        // must accept everything AzureContentSafetyConfig's timeout_ms
        // (u32) accepts, INCLUDING u32::MAX. A future edit that tightens
        // the schema below u32::MAX would make the loader stricter than
        // the struct and silently drop valid rows — this test fails loud.
        let v = json!({
            "name": "g",
            "kind": "azure_content_safety",
            "endpoint": "https://r.cognitiveservices.azure.com",
            "api_key": "k",
            "timeout_ms": 4_294_967_295u64
        });
        validate_guardrail(&v).unwrap();
    }

    #[test]
    fn guardrail_azure_content_safety_timeout_overflow_rejected() {
        // u32::MAX + 1 — beyond what the struct can deserialize. The
        // schema must reject it at the gate so the loader skips the row
        // cleanly instead of surfacing an opaque serde error downstream.
        let v = json!({
            "name": "g",
            "kind": "azure_content_safety",
            "endpoint": "https://r.cognitiveservices.azure.com",
            "api_key": "k",
            "timeout_ms": 4_294_967_296u64
        });
        assert!(validate_guardrail(&v).is_err());
    }

    #[test]
    fn guardrail_aliyun_text_moderation_passes() {
        // Minimal row: region + access keys. Optional fields (endpoint,
        // threshold, streaming params) omitted — the struct applies defaults.
        let v = json!({
            "name": "aliyun-guard",
            "kind": "aliyun_text_moderation",
            "hook_point": "both",
            "region": "cn-shanghai",
            "access_key_id": "LTAI_EXAMPLE",
            "access_key_secret": "plaintext-secret"
        });
        validate_guardrail(&v).unwrap();
    }

    #[test]
    fn guardrail_aliyun_text_moderation_with_optional_fields_passes() {
        let v = json!({
            "name": "aliyun-guard",
            "kind": "aliyun_text_moderation",
            "region": "cn-beijing",
            "endpoint": "http://127.0.0.1:8080",
            "access_key_id": "id",
            "access_key_secret": "secret",
            "risk_level_threshold": "medium",
            "timeout_ms": 3000,
            "stream_processing_mode": "buffer_full"
        });
        validate_guardrail(&v).unwrap();
    }

    #[test]
    fn guardrail_aliyun_text_moderation_missing_secret_rejected() {
        let v = json!({
            "name": "g",
            "kind": "aliyun_text_moderation",
            "region": "cn-shanghai",
            "access_key_id": "id"
        });
        assert!(validate_guardrail(&v).is_err());
    }

    #[test]
    fn guardrail_aliyun_text_moderation_bad_threshold_rejected() {
        let v = json!({
            "name": "g",
            "kind": "aliyun_text_moderation",
            "region": "cn-shanghai",
            "access_key_id": "id",
            "access_key_secret": "s",
            "risk_level_threshold": "none"
        });
        assert!(validate_guardrail(&v).is_err());
    }

    #[test]
    fn guardrail_rejects_invalid_string_enums() {
        let cases = [
            json!({
                "name": "g",
                "kind": "azure_content_safety_text_moderation",
                "endpoint": "https://example.cognitiveservices.azure.com",
                "api_key": "key",
                "output_type": "TwoSeverityLevels"
            }),
            json!({
                "name": "g",
                "kind": "azure_content_safety_text_moderation",
                "endpoint": "https://example.cognitiveservices.azure.com",
                "api_key": "key",
                "categories": ["Hate", "Spam"]
            }),
            json!({
                "name": "g",
                "kind": "azure_content_safety_text_moderation",
                "endpoint": "https://example.cognitiveservices.azure.com",
                "api_key": "key",
                "text_source": "assistant_only"
            }),
            json!({
                "name": "g",
                "kind": "azure_content_safety_text_moderation",
                "endpoint": "https://example.cognitiveservices.azure.com",
                "api_key": "key",
                "stream_processing_mode": "chunked"
            }),
            json!({
                "name": "g",
                "kind": "pii",
                "detectors": [{ "type": "email", "action": "redact" }]
            }),
            json!({
                "name": "g",
                "kind": "pii",
                "custom_patterns": [{
                    "name": "employee_id",
                    "regex": "\\bEMP-\\d+\\b",
                    "action": "redact"
                }]
            }),
            json!({
                "name": "g",
                "kind": "pii",
                "default_action": "redact",
                "detectors": [{ "type": "email" }]
            }),
            json!({
                "name": "g",
                "kind": "pii",
                "detectors": [{ "type": "driver_license" }]
            }),
            json!({
                "name": "g",
                "kind": "presidio",
                "analyzer_url": "http://presidio-analyzer:3000",
                "anonymizer_url": "http://presidio-anonymizer:3000",
                "default_action": "redact"
            }),
            json!({
                "name": "g",
                "kind": "presidio",
                "analyzer_url": "http://presidio-analyzer:3000",
                "anonymizer_url": "http://presidio-anonymizer:3000",
                "entities": [{ "type": "EMAIL_ADDRESS", "action": "redact" }]
            }),
            json!({
                "name": "g",
                "kind": "presidio",
                "analyzer_url": "http://presidio-analyzer:3000",
                "anonymizer_url": "http://presidio-anonymizer:3000",
                "operator": "tokenize"
            }),
        ];

        for value in cases {
            assert!(
                validate_guardrail(&value).is_err(),
                "guardrail schema accepted invalid enum value: {value}",
            );
        }
    }

    #[test]
    fn guardrail_fail_safe_fields_stay_schema_open() {
        let cases = [
            json!({
                "name": "g",
                "kind": "keyword",
                "patterns": [],
                "enforcement_mode": "audit",
                "direction": "sideways"
            }),
            json!({
                "name": "g",
                "kind": "azure_content_safety_text_moderation",
                "endpoint": "https://example.cognitiveservices.azure.com",
                "api_key": "key",
                "on_buffer_exceeded": "drop"
            }),
            json!({
                "name": "g",
                "kind": "aliyun_text_moderation",
                "region": "cn-shanghai",
                "access_key_id": "id",
                "access_key_secret": "s",
                "on_buffer_exceeded": "drop"
            }),
            json!({
                "name": "g",
                "kind": "pii",
                "detectors": [{ "type": "email" }],
                "on_buffer_exceeded": "drop"
            }),
            json!({
                "name": "g",
                "kind": "lakera",
                "api_key": "key",
                "on_buffer_exceeded": "drop"
            }),
            json!({
                "name": "g",
                "kind": "presidio",
                "analyzer_url": "http://presidio-analyzer:3000",
                "anonymizer_url": "http://presidio-anonymizer:3000",
                "on_buffer_exceeded": "drop"
            }),
        ];

        for value in cases {
            validate_guardrail(&value).unwrap();
        }
    }

    #[test]
    fn guardrail_openai_moderation_model_stays_open() {
        let v = json!({
            "name": "openai-mod",
            "kind": "openai_moderation",
            "api_key": "plaintext-key",
            "model": "future-moderation-model"
        });
        validate_guardrail(&v).unwrap();
    }

    // ---- observability_exporter schema tests ----

    #[test]
    fn exporter_otlp_http_happy_path() {
        let v = json!({
            "name": "honeycomb",
            "kind": "otlp_http",
            "endpoint": "https://api.honeycomb.io/v1/traces",
            "headers": { "x-honeycomb-team": "abc" }
        });
        validate_observability_exporter(&v).unwrap();
    }

    #[test]
    fn exporter_otlp_http_rejects_plain_http_endpoint() {
        let v = json!({
            "name": "x",
            "kind": "otlp_http",
            "endpoint": "http://api.honeycomb.io/v1/traces"
        });
        assert!(validate_observability_exporter(&v).is_err());
    }

    #[test]
    fn exporter_otlp_http_accepts_in_range_knobs() {
        // #519 B.2: sampling + content capture are real per-exporter knobs.
        for rate in [0.0, 0.5, 1.0] {
            let v = json!({
                "name": "otlp-knobs",
                "kind": "otlp_http",
                "endpoint": "https://api.honeycomb.io/v1/traces",
                "sample_rate": rate,
                "content_mode": "full",
                "content_max_bytes": 4096
            });
            validate_observability_exporter(&v).unwrap();
        }
    }

    #[test]
    fn exporter_otlp_http_rejects_out_of_range_sample_rate() {
        for rate in [-0.1, 1.1, 2.0] {
            let v = json!({
                "name": "x",
                "kind": "otlp_http",
                "endpoint": "https://api.honeycomb.io/v1/traces",
                "sample_rate": rate
            });
            assert!(
                validate_observability_exporter(&v).is_err(),
                "sample_rate {rate} must be rejected"
            );
        }
    }

    #[test]
    fn exporter_aliyun_sls_happy_path() {
        let v = json!({
            "name": "sls-prod",
            "kind": "aliyun_sls",
            "endpoint": "ap-southeast-3.log.aliyuncs.com",
            "project": "aisix-obs",
            "logstore": "request-events",
            "credential_ref": "sls-prod"
        });
        validate_observability_exporter(&v).unwrap();
    }

    #[test]
    fn exporter_aliyun_sls_allows_loopback_mock_endpoint() {
        // The L2 e2e points the DP at a local mock SLS over http://.
        let v = json!({
            "name": "sls-e2e",
            "kind": "aliyun_sls",
            "endpoint": "http://mock-sls:9000",
            "project": "p",
            "logstore": "l",
            "credential_ref": "mock"
        });
        validate_observability_exporter(&v).unwrap();
    }

    #[test]
    fn exporter_object_store_happy_path() {
        let v = json!({
            "name": "acme-s3",
            "kind": "object_store",
            "provider": "s3",
            "bucket": "acme-aisix-events",
            "prefix": "ai-gateway",
            "region": "us-east-1",
            "credential_ref": "acme-s3"
        });
        validate_observability_exporter(&v).unwrap();
    }

    #[test]
    fn exporter_object_store_requires_core_fields() {
        // Each config missing one required object_store field is rejected.
        let cases = [
            json!({"name":"x","kind":"object_store","bucket":"b","prefix":"p","credential_ref":"r"}),
            json!({"name":"x","kind":"object_store","provider":"s3","prefix":"p","credential_ref":"r"}),
            json!({"name":"x","kind":"object_store","provider":"s3","bucket":"b","credential_ref":"r"}),
            json!({"name":"x","kind":"object_store","provider":"s3","bucket":"b","prefix":"p"}),
        ];
        for v in cases {
            assert!(
                validate_observability_exporter(&v).is_err(),
                "incomplete object_store config must be rejected: {v}"
            );
        }
    }

    #[test]
    fn exporter_object_store_rejects_bad_provider() {
        let v = json!({
            "name": "x", "kind": "object_store",
            "provider": "wasabi", "bucket": "b", "prefix": "p", "credential_ref": "r"
        });
        assert!(validate_observability_exporter(&v).is_err());
    }

    #[test]
    fn exporter_object_store_cloud_identity_omits_credential_ref() {
        // cloud_identity (S3 / GCS): the DP uses its own attached identity, so
        // credential_ref is NOT required.
        for provider in ["s3", "gcs"] {
            let v = json!({
                "name": "x", "kind": "object_store",
                "provider": provider, "bucket": "b", "prefix": "p",
                "auth_mode": "cloud_identity"
            });
            validate_observability_exporter(&v)
                .unwrap_or_else(|e| panic!("cloud_identity {provider} should validate: {e:?}"));
        }
    }

    #[test]
    fn exporter_object_store_cloud_identity_rejects_azure() {
        // Azure cloud_identity is unsupported (managed identity needs a
        // non-secret account name the keyless config does not carry).
        let v = json!({
            "name": "x", "kind": "object_store",
            "provider": "azure_blob", "bucket": "c", "prefix": "p",
            "auth_mode": "cloud_identity"
        });
        assert!(validate_observability_exporter(&v).is_err());
    }

    #[test]
    fn exporter_object_store_credential_ref_mode_still_requires_ref() {
        // Default (no auth_mode) and explicit credential_ref both require the
        // ref — only cloud_identity drops it.
        for v in [
            json!({"name":"x","kind":"object_store","provider":"s3","bucket":"b","prefix":"p"}),
            json!({"name":"x","kind":"object_store","provider":"s3","bucket":"b","prefix":"p","auth_mode":"credential_ref"}),
        ] {
            assert!(
                validate_observability_exporter(&v).is_err(),
                "credential_ref must be required outside cloud_identity: {v}"
            );
        }
    }

    #[test]
    fn exporter_object_store_allows_loopback_minio_endpoint() {
        // The e2e points the S3 sink at a local MinIO over http://.
        let v = json!({
            "name": "s3-e2e", "kind": "object_store",
            "provider": "s3", "bucket": "b", "prefix": "p",
            "endpoint": "http://minio:9000", "credential_ref": "mock"
        });
        validate_observability_exporter(&v).unwrap();
    }

    #[test]
    fn exporter_object_store_rejects_plaintext_non_loopback_endpoint() {
        // A non-loopback plaintext endpoint must be rejected — no exfil to an
        // arbitrary http host.
        let v = json!({
            "name": "x", "kind": "object_store",
            "provider": "s3", "bucket": "b", "prefix": "p",
            "endpoint": "http://evil.example.com", "credential_ref": "r"
        });
        assert!(validate_observability_exporter(&v).is_err());
    }

    #[test]
    fn exporter_aliyun_sls_requires_project_logstore_credential() {
        for missing in ["project", "logstore", "credential_ref"] {
            let mut v = json!({
                "name": "x",
                "kind": "aliyun_sls",
                "endpoint": "ap-southeast-3.log.aliyuncs.com",
                "project": "p",
                "logstore": "l",
                "credential_ref": "r"
            });
            v.as_object_mut().unwrap().remove(missing);
            assert!(
                validate_observability_exporter(&v).is_err(),
                "missing `{missing}` must be rejected"
            );
        }
    }

    #[test]
    fn exporter_aliyun_sls_rejects_plaintext_credentials() {
        // No AccessKey field is allowed at the schema layer either —
        // `additionalProperties: false` rejects it before serde runs.
        let v = json!({
            "name": "x",
            "kind": "aliyun_sls",
            "endpoint": "ap-southeast-3.log.aliyuncs.com",
            "project": "p",
            "logstore": "l",
            "credential_ref": "r",
            "access_key_secret": "AKIASECRET"
        });
        assert!(validate_observability_exporter(&v).is_err());
    }

    #[test]
    fn exporter_aliyun_sls_content_capture_fields() {
        let base = |extra: serde_json::Value| {
            let mut v = json!({
                "name": "x",
                "kind": "aliyun_sls",
                "endpoint": "ap-southeast-3.log.aliyuncs.com",
                "project": "p",
                "logstore": "l",
                "credential_ref": "r"
            });
            let obj = v.as_object_mut().unwrap();
            for (k, val) in extra.as_object().unwrap() {
                obj.insert(k.clone(), val.clone());
            }
            v
        };
        // Opt-in content capture validates.
        validate_observability_exporter(&base(
            json!({ "content_mode": "full", "content_max_bytes": 4096 }),
        ))
        .unwrap();
        // Unknown content_mode is rejected.
        assert!(
            validate_observability_exporter(&base(json!({ "content_mode": "verbose" }))).is_err()
        );
        // content_max_bytes must be a positive integer.
        assert!(validate_observability_exporter(&base(json!({ "content_max_bytes": 0 }))).is_err());
    }

    #[test]
    fn exporter_rejects_unknown_kind() {
        let v = json!({ "name": "x", "kind": "splunk_hec", "endpoint": "https://x" });
        assert!(validate_observability_exporter(&v).is_err());
    }

    #[test]
    fn exporter_datadog_happy_path() {
        let v = json!({
            "name": "datadog-prod",
            "kind": "datadog",
            "site": "datadoghq.com",
            "credential_ref": "datadog-prod",
            "service": "ai-gateway",
            "ddsource": "aisix-ai-gateway",
            "tags": ["team:platform", "tier:prod"]
        });
        validate_observability_exporter(&v).unwrap();
    }

    #[test]
    fn exporter_datadog_accepts_every_allow_list_site() {
        for site in [
            "datadoghq.com",
            "us3.datadoghq.com",
            "us5.datadoghq.com",
            "datadoghq.eu",
            "ap1.datadoghq.com",
            "ap2.datadoghq.com",
            "ddog-gov.com",
        ] {
            let v = json!({
                "name": "x",
                "kind": "datadog",
                "site": site,
                "credential_ref": "r",
                "service": "s"
            });
            validate_observability_exporter(&v)
                .unwrap_or_else(|e| panic!("site {site:?} must validate: {e:?}"));
        }
    }

    #[test]
    fn exporter_datadog_rejects_non_allow_list_site() {
        // A plausible-looking but unsupported / spoofed site must be rejected —
        // no exfil to an arbitrary `http-intake.logs.<host>`.
        for bad in [
            "evil.datadoghq.com.attacker.test",
            "datadoghq.org",
            "us9.datadoghq.com",
            "datadog.com",
            "datadoghq.com:443", // a port is NOT allowed on a real site
            "",
        ] {
            let v = json!({
                "name": "x",
                "kind": "datadog",
                "site": bad,
                "credential_ref": "r",
                "service": "s"
            });
            assert!(
                validate_observability_exporter(&v).is_err(),
                "site {bad:?} must be rejected by the allow-list"
            );
        }
    }

    #[test]
    fn exporter_datadog_allows_loopback_mock_site() {
        // The e2e points the DP at a local mock Datadog intake — bare host OR
        // host:port. The harness binds a FREE port, so `:port` must validate
        // (the prior exact-enum rejected it while the sink accepted it — #548).
        for site in ["mock-datadog", "127.0.0.1:54321", "localhost:8080"] {
            let v = json!({
                "name": "datadog-e2e",
                "kind": "datadog",
                "site": site,
                "credential_ref": "mock",
                "service": "ai-gateway"
            });
            validate_observability_exporter(&v)
                .unwrap_or_else(|e| panic!("loopback site {site:?} must validate: {e:?}"));
        }
    }

    #[test]
    fn exporter_datadog_requires_site_credential_service() {
        for missing in ["site", "credential_ref", "service"] {
            let mut v = json!({
                "name": "x",
                "kind": "datadog",
                "site": "datadoghq.com",
                "credential_ref": "r",
                "service": "s"
            });
            v.as_object_mut().unwrap().remove(missing);
            assert!(
                validate_observability_exporter(&v).is_err(),
                "missing `{missing}` must be rejected"
            );
        }
    }

    #[test]
    fn exporter_datadog_rejects_plaintext_api_key() {
        // No API-key field is allowed at the schema layer either —
        // `additionalProperties: false` rejects it before serde runs.
        let v = json!({
            "name": "x",
            "kind": "datadog",
            "site": "datadoghq.com",
            "credential_ref": "r",
            "service": "s",
            "api_key": "DDSECRET"
        });
        assert!(validate_observability_exporter(&v).is_err());
    }

    #[test]
    fn exporter_datadog_content_capture_fields() {
        let base = |extra: serde_json::Value| {
            let mut v = json!({
                "name": "x",
                "kind": "datadog",
                "site": "datadoghq.com",
                "credential_ref": "r",
                "service": "s"
            });
            let obj = v.as_object_mut().unwrap();
            for (k, val) in extra.as_object().unwrap() {
                obj.insert(k.clone(), val.clone());
            }
            v
        };
        // Opt-in content capture validates.
        validate_observability_exporter(&base(
            json!({ "content_mode": "full", "content_max_bytes": 4096 }),
        ))
        .unwrap();
        // Unknown content_mode is rejected.
        assert!(
            validate_observability_exporter(&base(json!({ "content_mode": "verbose" }))).is_err()
        );
        // content_max_bytes must be a positive integer (min 1).
        assert!(validate_observability_exporter(&base(json!({ "content_max_bytes": 0 }))).is_err());
        // content_max_bytes is capped at 1 MiB (Datadog per-log limit).
        assert!(
            validate_observability_exporter(&base(json!({ "content_max_bytes": 1_048_577 })))
                .is_err()
        );
    }

    // ---- rate_limit_policy schema tests ----

    #[test]
    fn rate_limit_policy_happy_path() {
        let v = json!({
            "name": "team-quota",
            "scope": "team",
            "scope_ref": "team-uuid-1",
            "window": "minute",
            "max_requests": 100,
            "max_tokens": 50000
        });
        validate_rate_limit_policy(&v).unwrap();
    }

    #[test]
    fn rate_limit_policy_rejects_unknown_scope() {
        let v = json!({
            "name": "bad",
            "scope": "org",
            "scope_ref": "x",
            "window": "minute",
            "max_requests": 10
        });
        assert!(validate_rate_limit_policy(&v).is_err());
    }

    #[test]
    fn rate_limit_policy_rejects_unknown_window() {
        let v = json!({
            "name": "bad",
            "scope": "team",
            "scope_ref": "x",
            "window": "day",
            "max_requests": 10
        });
        assert!(validate_rate_limit_policy(&v).is_err());
    }

    #[test]
    fn rate_limit_policy_rejects_extra_field() {
        let v = json!({
            "name": "bad",
            "scope": "team",
            "scope_ref": "x",
            "window": "minute",
            "max_requests": 10,
            "extra": 1
        });
        assert!(validate_rate_limit_policy(&v).is_err());
    }

    #[test]
    fn rate_limit_policy_rejects_zero_max_requests() {
        let v = json!({
            "name": "bad",
            "scope": "team",
            "scope_ref": "x",
            "window": "minute",
            "max_requests": 0
        });
        assert!(validate_rate_limit_policy(&v).is_err());
    }

    #[test]
    fn rate_limit_policy_rejects_no_limits() {
        let v = json!({
            "name": "noop",
            "scope": "team",
            "scope_ref": "x",
            "window": "minute"
        });
        assert!(validate_rate_limit_policy(&v).is_err());
    }

    // ---- provider_key schema (issue #302 Phase A skeleton) ----

    #[test]
    fn provider_key_minimal_passes() {
        let v = json!({
            "display_name": "openai-prod",
            "secret": "sk-x"
        });
        validate_provider_key(&v).unwrap();
    }

    #[test]
    fn provider_key_legacy_payload_without_phase_a_fields_passes() {
        // Pre-#302 payload — no provider / adapter / telemetry_tags.
        // Must still validate so existing on-disk rows keep loading.
        let v = json!({
            "display_name": "openai-prod",
            "secret": "sk-x",
            "api_base": "https://api.openai.com/v1"
        });
        validate_provider_key(&v).unwrap();
    }

    #[test]
    fn provider_key_with_phase_a_fields_passes() {
        let v = json!({
            "display_name": "deepseek-prod",
            "secret": "sk-x",
            "api_base": "https://api.deepseek.com/v1",
            "provider": "deepseek",
            "adapter": "openai",
            "telemetry_tags": {
                "kind": "catalog",
                "featured": true,
                "branded_provider": "deepseek",
                "pk_label": "production"
            }
        });
        validate_provider_key(&v).unwrap();
    }

    #[test]
    fn provider_key_with_byo_telemetry_shape_passes() {
        let v = json!({
            "display_name": "internal-llm",
            "secret": "sk-x",
            "telemetry_tags": {
                "kind": "byo",
                "branded_provider": null,
                "byo_label": "platform-team"
            }
        });
        validate_provider_key(&v).unwrap();
    }

    #[test]
    fn provider_key_rejects_unknown_adapter_value() {
        let v = json!({
            "display_name": "x",
            "secret": "k",
            "adapter": "not-a-real-adapter"
        });
        assert!(validate_provider_key(&v).is_err());
    }

    #[test]
    fn provider_key_rejects_unknown_telemetry_field() {
        let v = json!({
            "display_name": "x",
            "secret": "k",
            "telemetry_tags": { "unknown_tag": "v" }
        });
        assert!(validate_provider_key(&v).is_err());
    }

    #[test]
    fn provider_key_rejects_unknown_top_level_field() {
        // Top-level additionalProperties=false still applies — only
        // the explicitly-listed Phase A fields are accepted.
        let v = json!({
            "display_name": "x",
            "secret": "k",
            "rogue": 1
        });
        assert!(validate_provider_key(&v).is_err());
    }

    #[test]
    fn provider_key_rejects_unknown_telemetry_kind() {
        // `kind` is the closed `"catalog" | "byo"` set.
        let v = json!({
            "display_name": "x",
            "secret": "k",
            "telemetry_tags": { "kind": "third-party" }
        });
        assert!(validate_provider_key(&v).is_err());
    }

    // ---- provider_key schema (issue #302 Phase A2.5 — request/response) ----

    #[test]
    fn provider_key_with_request_block_passes() {
        // Mirror the on-disk example in issue #302 §5 exactly.
        let v = json!({
            "display_name": "deepseek-prod",
            "secret": "sk-x",
            "request": {
                "param_renames":       { "max_completion_tokens": "max_tokens" },
                "param_constraints":   { "temperature_max": 1.0 },
                "default_headers":     { "X-Foo": "bar" },
                "default_body_fields": { "safe_prompt": true }
            }
        });
        validate_provider_key(&v).unwrap();
    }

    #[test]
    fn provider_key_with_response_block_passes() {
        let v = json!({
            "display_name": "deepseek-prod",
            "secret": "sk-x",
            "response": {
                "stream_done_marker":     "required",
                "content_list_to_string": false,
                "error_envelope":         "openai",
                "reasoning_field":        "delta.reasoning_content"
            }
        });
        validate_provider_key(&v).unwrap();
    }

    #[test]
    fn provider_key_with_empty_request_response_blocks_passes() {
        // `{}` for each block must validate — matches the Rust-side
        // all-default deserialization path.
        let v = json!({
            "display_name": "x",
            "secret": "k",
            "request": {},
            "response": {}
        });
        validate_provider_key(&v).unwrap();
    }

    #[test]
    fn provider_key_request_rejects_unknown_field() {
        let v = json!({
            "display_name": "x",
            "secret": "k",
            "request": { "param_rename": {} }
        });
        assert!(validate_provider_key(&v).is_err());
    }

    #[test]
    fn provider_key_response_rejects_unknown_field() {
        let v = json!({
            "display_name": "x",
            "secret": "k",
            "response": { "reasoning_fields": "delta.foo" }
        });
        assert!(validate_provider_key(&v).is_err());
    }

    #[test]
    fn provider_key_response_rejects_unknown_stream_done_marker() {
        let v = json!({
            "display_name": "x",
            "secret": "k",
            "response": { "stream_done_marker": "maybe" }
        });
        assert!(validate_provider_key(&v).is_err());
    }

    #[test]
    fn provider_key_request_param_constraints_rejects_unknown_field() {
        // `param_constraints` is closed (`additionalProperties: false`)
        // so a stray `top_p_max` from a future schema iteration can't
        // sneak past today's DP.
        let v = json!({
            "display_name": "x",
            "secret": "k",
            "request": {
                "param_constraints": { "top_p_max": 0.9 }
            }
        });
        assert!(validate_provider_key(&v).is_err());
    }

    // ---- mcp_server schema tests (#666 timeout_ms guard) ----

    #[test]
    fn mcp_server_minimal_passes() {
        // `timeout_ms` is optional; omitting it must validate.
        let v = json!({
            "display_name": "github",
            "url": "https://api.example.com/mcp"
        });
        validate_mcp_server(&v).unwrap();
    }

    #[test]
    fn mcp_server_accepts_positive_timeout_ms() {
        let v = json!({
            "display_name": "github",
            "url": "https://api.example.com/mcp",
            "timeout_ms": 1
        });
        validate_mcp_server(&v).unwrap();
    }

    #[test]
    fn mcp_server_accepts_api_key_auth() {
        let v = json!({
            "display_name": "github",
            "url": "https://api.example.com/mcp",
            "auth_type": "api_key",
            "secret": "k-123"
        });
        validate_mcp_server(&v).unwrap();
    }

    #[test]
    fn mcp_server_accepts_oauth2_auth_with_client_fields() {
        let v = json!({
            "display_name": "github",
            "url": "https://api.example.com/mcp",
            "auth_type": "oauth2",
            "secret": "client-secret",
            "client_id": "cid",
            "token_url": "https://auth.example.com/oauth/token",
            "scopes": ["read", "write"]
        });
        validate_mcp_server(&v).unwrap();
    }

    #[test]
    fn mcp_server_rejects_unknown_auth_type_and_bad_scopes_shape() {
        // The `auth_type` set is closed: near-misses like `oauth` must fail.
        let v = json!({
            "display_name": "x",
            "url": "https://x/mcp",
            "auth_type": "oauth"
        });
        assert!(validate_mcp_server(&v).is_err());

        // `scopes` is an array of strings, not a single space-joined string.
        let v = json!({
            "display_name": "x",
            "url": "https://x/mcp",
            "auth_type": "oauth2",
            "secret": "s",
            "client_id": "cid",
            "token_url": "https://auth/token",
            "scopes": "read write"
        });
        assert!(validate_mcp_server(&v).is_err());
    }

    #[test]
    fn mcp_server_schema_stays_permissive_on_credential_coupling() {
        // The per-`auth_type` credential coupling (oauth2 ⇒ client_id +
        // secret + token_url) is enforced by write paths, not this schema —
        // an incomplete oauth2 row must still validate so the snapshot loader
        // keeps it (the runtime degrades that server gracefully instead).
        let v = json!({
            "display_name": "x",
            "url": "https://x/mcp",
            "auth_type": "oauth2"
        });
        validate_mcp_server(&v).unwrap();
    }

    #[test]
    fn mcp_server_rejects_zero_timeout_ms() {
        // A zero deadline times out every upstream op instantly and silently
        // drops the server from `tools/list`; reject it at the schema layer
        // (enforced on both the Admin write-path and the etcd loader).
        let v = json!({
            "display_name": "github",
            "url": "https://api.example.com/mcp",
            "timeout_ms": 0
        });
        assert!(validate_mcp_server(&v).is_err());
    }
}
