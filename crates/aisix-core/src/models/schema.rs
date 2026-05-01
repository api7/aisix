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
    pub credential: Validator,
    pub team: Validator,
    pub guardrail: Validator,
}

pub static SCHEMAS: Lazy<Arc<Schemas>> = Lazy::new(|| Arc::new(Schemas::compile()));

impl Schemas {
    fn compile() -> Self {
        Self {
            model: jsonschema::options()
                .build(&model_schema())
                .expect("model schema is well-formed"),
            apikey: jsonschema::options()
                .build(&apikey_schema())
                .expect("apikey schema is well-formed"),
            credential: jsonschema::options()
                .build(&credential_schema())
                .expect("credential schema is well-formed"),
            team: jsonschema::options()
                .build(&team_schema())
                .expect("team schema is well-formed"),
            guardrail: jsonschema::options()
                .build(&guardrail_schema())
                .expect("guardrail schema is well-formed"),
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

pub fn validate_credential(value: &Value) -> Result<(), SchemaError> {
    validate(&SCHEMAS.credential, value)
}

pub fn validate_team(value: &Value) -> Result<(), SchemaError> {
    validate(&SCHEMAS.team, value)
}

pub fn validate_guardrail(value: &Value) -> Result<(), SchemaError> {
    validate(&SCHEMAS.guardrail, value)
}

fn model_schema() -> Value {
    json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["name", "model", "provider_config"],
        "additionalProperties": false,
        "properties": {
            "name": { "type": "string", "minLength": 1 },
            "model": {
                "type": "string",
                "pattern": "^(anthropic|deepseek|gemini|openai|router)/.+$"
            },
            "provider_config": {
                "type": "object",
                "required": ["api_key"],
                "additionalProperties": false,
                "properties": {
                    "api_key": { "type": "string", "minLength": 1 },
                    "api_base": { "type": "string" }
                }
            },
            "timeout": { "type": "integer", "minimum": 0 },
            "rate_limit": { "$ref": "#/$defs/rate_limit" },
            "routing": {
                "type": "object",
                "required": ["targets"],
                "additionalProperties": false,
                "properties": {
                    "strategy": {
                        "type": "string",
                        "enum": ["round_robin", "weighted", "failover"]
                    },
                    "targets": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "required": ["model"],
                            "additionalProperties": false,
                            "properties": {
                                "model":  { "type": "string", "minLength": 1 },
                                "weight": { "type": "integer", "minimum": 0 }
                            }
                        }
                    },
                    "retry_budget": { "type": "integer", "minimum": 0 }
                }
            }
        },
        "$defs": {
            "rate_limit": {
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "tpm":         { "type": "integer", "minimum": 0 },
                    "tpd":         { "type": "integer", "minimum": 0 },
                    "rpm":         { "type": "integer", "minimum": 0 },
                    "rpd":         { "type": "integer", "minimum": 0 },
                    "concurrency": { "type": "integer", "minimum": 0 }
                }
            }
        }
    })
}

fn apikey_schema() -> Value {
    json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["key_hash", "allowed_models"],
        "additionalProperties": false,
        "properties": {
            "key_hash": { "type": "string", "minLength": 1 },
            "allowed_models": {
                "type": "array",
                "items": { "type": "string" }
            },
            "rate_limit": { "$ref": "#/$defs/rate_limit" }
        },
        "$defs": {
            "rate_limit": {
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "tpm":         { "type": "integer", "minimum": 0 },
                    "tpd":         { "type": "integer", "minimum": 0 },
                    "rpm":         { "type": "integer", "minimum": 0 },
                    "rpd":         { "type": "integer", "minimum": 0 },
                    "concurrency": { "type": "integer", "minimum": 0 }
                }
            }
        }
    })
}

fn credential_schema() -> Value {
    json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["name", "api_key"],
        "additionalProperties": false,
        "properties": {
            "name":     { "type": "string", "minLength": 1 },
            "api_key":  { "type": "string", "minLength": 1 },
            "api_base": { "type": "string" }
        }
    })
}

fn team_schema() -> Value {
    json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["name"],
        "additionalProperties": false,
        "properties": {
            "name":      { "type": "string", "minLength": 1 },
            "members":   {
                "type": "array",
                "items": { "type": "string", "minLength": 1 }
            },
            "budget_id": { "type": "string", "minLength": 1 },
            "rate_limit": { "$ref": "#/$defs/rate_limit" }
        },
        "$defs": {
            "rate_limit": {
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "tpm":         { "type": "integer", "minimum": 0 },
                    "tpd":         { "type": "integer", "minimum": 0 },
                    "rpm":         { "type": "integer", "minimum": 0 },
                    "rpd":         { "type": "integer", "minimum": 0 },
                    "concurrency": { "type": "integer", "minimum": 0 }
                }
            }
        }
    })
}

fn guardrail_schema() -> Value {
    json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["name", "kind"],
        // Each kind variant adds its own keys; the per-kind oneOf
        // below pins them. Top-level stays open so future kinds
        // (lakera, protect_ai) only edit the oneOf branch.
        "additionalProperties": true,
        "properties": {
            "name":       { "type": "string", "minLength": 1 },
            "enabled":    { "type": "boolean" },
            "hook_point": { "enum": ["input", "output", "both"] },
            "fail_open":  { "type": "boolean" },
            "kind":       { "enum": ["keyword", "bedrock"] }
        },
        "oneOf": [
            {
                "type": "object",
                "required": ["kind", "patterns"],
                "properties": {
                    "kind":     { "const": "keyword" },
                    "patterns": {
                        "type": "array",
                        "items": { "$ref": "#/$defs/keyword_pattern" }
                    }
                }
            },
            {
                "type": "object",
                "required": [
                    "kind", "guardrail_id", "guardrail_version",
                    "region", "aws_credentials", "latency_mode"
                ],
                "properties": {
                    "kind":               { "const": "bedrock" },
                    "guardrail_id":       { "type": "string", "minLength": 1, "maxLength": 64 },
                    "guardrail_version":  { "type": "string", "minLength": 1, "maxLength": 16 },
                    "region":             { "type": "string", "minLength": 1 },
                    "aws_credentials":    { "$ref": "#/$defs/bedrock_aws_credentials" },
                    "latency_mode":       { "$ref": "#/$defs/bedrock_latency_mode" }
                }
            }
        ],
        "$defs": {
            "keyword_pattern": {
                "type": "object",
                "additionalProperties": false,
                "required": ["kind", "value"],
                "properties": {
                    "kind":  { "enum": ["literal", "regex"] },
                    "value": { "type": "string", "minLength": 1 }
                }
            },
            "bedrock_aws_credentials": {
                "type": "object",
                // v1 ships kind=static (plaintext access keys on the
                // kine wire — cp-api decrypts the envelope-encrypted
                // secret at projection time, see PRD-09c §6.3).
                // Phase 4 adds kind=role_arn (sts:AssumeRole) under
                // the same `kind` discriminator.
                "required": ["kind", "access_key_id", "secret_access_key"],
                "properties": {
                    "kind":              { "const": "static" },
                    "access_key_id":     { "type": "string", "minLength": 1 },
                    "secret_access_key": { "type": "string", "minLength": 1 }
                },
                "additionalProperties": false
            },
            "bedrock_latency_mode": {
                "oneOf": [
                    {
                        "type": "object",
                        "additionalProperties": false,
                        "required": ["kind"],
                        "properties": { "kind": { "const": "serial" } }
                    },
                    {
                        "type": "object",
                        "additionalProperties": false,
                        "required": ["kind", "timeout_ms"],
                        "properties": {
                            "kind":       { "const": "timed" },
                            "timeout_ms": { "type": "integer", "minimum": 100, "maximum": 5000 }
                        }
                    }
                ]
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn model_happy_path_passes() {
        let v = json!({
            "name": "my-gpt4",
            "model": "openai/gpt-4o",
            "provider_config": {"api_key": "sk-x"},
            "timeout": 30000,
            "rate_limit": {"rpm": 100}
        });
        validate_model(&v).unwrap();
    }

    #[test]
    fn model_missing_name_fails_with_useful_path() {
        let v = json!({
            "model": "openai/gpt-4o",
            "provider_config": {"api_key": "sk-x"}
        });
        let err = validate_model(&v).unwrap_err();
        assert!(err.message.contains("name"));
    }

    #[test]
    fn model_bad_provider_prefix_fails() {
        let v = json!({
            "name": "x",
            "model": "mistral/large",
            "provider_config": {"api_key": "k"}
        });
        let err = validate_model(&v).unwrap_err();
        assert!(err.path.contains("/model") || err.message.to_lowercase().contains("pattern"));
    }

    #[test]
    fn model_rejects_additional_top_level() {
        let v = json!({
            "name":"x","model":"openai/g","provider_config":{"api_key":"k"},
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
    fn rate_limit_negative_value_rejected() {
        let v = json!({
            "name":"x","model":"openai/g","provider_config":{"api_key":"k"},
            "rate_limit": {"rpm": -1}
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
}
