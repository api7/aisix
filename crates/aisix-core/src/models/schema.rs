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
    pub cache_policy: Validator,
    pub observability_exporter: Validator,
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
            provider_key: jsonschema::options()
                .build(&provider_key_schema())
                .expect("provider_key schema is well-formed"),
            guardrail: jsonschema::options()
                .build(&guardrail_schema())
                .expect("guardrail schema is well-formed"),
            cache_policy: jsonschema::options()
                .build(&cache_policy_schema())
                .expect("cache_policy schema is well-formed"),
            observability_exporter: jsonschema::options()
                .build(&observability_exporter_schema())
                .expect("observability_exporter schema is well-formed"),
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

fn model_schema() -> Value {
    json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["display_name"],
        "additionalProperties": false,
        "properties": {
            "display_name":    { "type": "string", "minLength": 1 },
            "provider":        { "type": "string", "enum": ["openai","anthropic","gemini","deepseek","cohere","jina"] },
            "model_name":      { "type": "string", "minLength": 1 },
            "provider_key_id": { "type": "string", "minLength": 1 },
            "timeout":         { "type": "integer", "minimum": 0 },
            "rate_limit":      { "$ref": "#/$defs/rate_limit" },
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
            },
            "cost": {
                "type": "object",
                "required": ["input_per_1k", "output_per_1k"],
                "additionalProperties": false,
                "properties": {
                    "input_per_1k":  { "type": "number", "minimum": 0 },
                    "output_per_1k": { "type": "number", "minimum": 0 }
                }
            }
        },
        // Direct vs routing model: a model EITHER ships a `routing`
        // block (virtual router — provider/model_name/provider_key_id
        // forbidden) OR ships those three required fields together
        // (direct upstream — routing forbidden).
        "oneOf": [
            {
                "required": ["routing"],
                "not": { "anyOf": [
                    { "required": ["provider"] },
                    { "required": ["model_name"] },
                    { "required": ["provider_key_id"] }
                ]}
            },
            {
                "required": ["provider", "model_name", "provider_key_id"],
                "not": { "required": ["routing"] }
            }
        ],
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

fn provider_key_schema() -> Value {
    json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["display_name", "secret"],
        "additionalProperties": false,
        "properties": {
            "display_name": { "type": "string", "minLength": 1 },
            "secret":       { "type": "string", "minLength": 1 },
            "api_base":     { "type": "string" }
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

// Mirrors cp-api's cache_policies validation rules (validateCachePolicyShape
// in internal/cpapi/resources/cache_policies.go). The DP is the second
// line of defence — cp-api rejects malformed payloads on write, but kine
// can still surface stale or hand-edited rows on watch, so we re-validate
// at parse time. `additionalProperties: true` keeps the schema
// forward-compatible: cp-api can ship new optional fields ahead of a DP
// rollout without locking the gateway out.
fn cache_policy_schema() -> Value {
    // Backends: memory + redis. Semantic backends were removed
    // pending DP-side wiring — see ai-gateway issue #116. The schema
    // stays `additionalProperties: true` so a newer cp-api can ship
    // forward-compat fields without locking out an older DP.
    json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["name"],
        "additionalProperties": true,
        "properties": {
            "name":        { "type": "string", "minLength": 1, "maxLength": 120 },
            "enabled":     { "type": "boolean" },
            "backend":     { "enum": ["memory", "redis"] },
            "ttl_seconds": { "type": "integer", "minimum": 1, "maximum": 604800 },
            "applies_to":  { "type": "string", "minLength": 1, "maxLength": 255 }
        }
    })
}

fn observability_exporter_schema() -> Value {
    // MVP: single discriminator value `otlp_http` whose fields land
    // flat at the top level (matches the Guardrail wire shape — see
    // `models/observability_exporter.rs` doc comment). Phase 2 adds
    // `helicone` / `datadog_logs` / `s3_ndjson` as additional
    // discriminator values with their own `if`/`then` branches below.
    json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["name", "kind"],
        "additionalProperties": false,
        "properties": {
            "name":    { "type": "string", "minLength": 1, "maxLength": 120 },
            "enabled": { "type": "boolean" },
            "kind":    { "type": "string", "enum": ["otlp_http"] },
            // otlp_http branch — flat fields.
            "endpoint": {
                "type": "string",
                // Reject http:// and any non-URL by anchoring on https://.
                // Loopback bypass for e2e: allow http://mock-otlp:* /
                // http://127.0.0.1 / http://localhost so the compose
                // test can wire a fake receiver without TLS.
                "pattern": "^https://.+|^http://(mock-otlp|otel-collector|127\\.0\\.0\\.1|localhost)(:[0-9]+)?(/.*)?$"
            },
            "headers": {
                "type": "object",
                "additionalProperties": { "type": "string" }
            }
        },
        "allOf": [
            {
                "if":   { "properties": { "kind": { "const": "otlp_http" } } },
                "then": { "required": ["endpoint"] }
            }
        ]
    })
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
    fn model_missing_display_name_fails() {
        let v = json!({
            "provider": "openai",
            "model_name": "gpt-4o",
            "provider_key_id": "pk-1"
        });
        let err = validate_model(&v).unwrap_err();
        assert!(err.message.to_lowercase().contains("display_name"));
    }

    #[test]
    fn model_unknown_provider_value_fails() {
        let v = json!({
            "display_name": "x",
            "provider": "mistral",
            "model_name": "large",
            "provider_key_id": "pk-1"
        });
        assert!(validate_model(&v).is_err());
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
    fn apikey_unknown_field_rejected() {
        let v = json!({
            "key_hash":"9df37f5e7cbc3c391d872742b5f286c242e733a09add9eeaa4d26a599bd90b20",
            "allowed_models":["a"],
            "max_budget_usd": 500.0
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
