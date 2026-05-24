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
    pub rate_limit_policy: Validator,
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
            rate_limit_policy: jsonschema::options()
                .build(&rate_limit_policy_schema())
                .expect("rate_limit_policy schema is well-formed"),
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

fn model_schema() -> Value {
    json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["display_name"],
        "additionalProperties": false,
        "properties": {
            "display_name":    { "type": "string", "minLength": 1 },
            // `provider` is the open vendor identity (models.dev catalog id
            // — e.g. `openai`, `xai`, `wafer.ai`). The pattern accepts the
            // dot character because at least one real models.dev id
            // (`wafer.ai`) contains it; rejecting `.` would re-create the
            // #417 bug class for that vendor.
            "provider":        { "type": "string", "minLength": 1, "maxLength": 64, "pattern": "^[a-z0-9][a-z0-9._-]*$" },
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
                    "retries": { "type": "integer", "minimum": 0 },
                    "max_fallbacks": { "type": "integer", "minimum": 0 },
                    "retry_on_429": { "type": "boolean" },
                    "on_all_filtered": {
                        "type": "string",
                        "enum": ["fail", "original_order"]
                    }
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
            },
            "background_model_check": {
                "type": "object",
                "required": [
                    "enabled",
                    "interval_seconds",
                    "timeout_seconds",
                    "prompt",
                    "max_tokens",
                    "stale_after_seconds"
                ],
                "additionalProperties": false,
                "properties": {
                    "enabled": { "type": "boolean" },
                    // Minimum 5s guards against misconfiguration. Setting
                    // interval_seconds=1 with multiple direct models would
                    // burn provider quota and money very quickly.
                    "interval_seconds": { "type": "integer", "minimum": 5 },
                    "timeout_seconds": { "type": "integer", "minimum": 1 },
                    "prompt": { "type": "string", "minLength": 1 },
                    "max_tokens": { "type": "integer", "minimum": 1 },
                    "ignore_statuses": {
                        "type": "array",
                        "items": { "type": "integer", "minimum": 100, "maximum": 599 }
                    },
                    "stale_after_seconds": { "type": "integer", "minimum": 1 }
                }
            },
            "cooldown": {
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "enabled":              { "type": "boolean" },
                    "default_seconds":      { "type": "integer", "minimum": 0 },
                    "max_seconds":          { "type": "integer", "minimum": 1 },
                    "honor_retry_after":    { "type": "boolean" },
                    "trigger_statuses": {
                        "type": "array",
                        "items": { "type": "integer", "minimum": 100, "maximum": 599 }
                    },
                    "trigger_on_timeout":   { "type": "boolean" },
                    "trigger_on_transport": { "type": "boolean" }
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
                    { "required": ["provider_key_id"] },
                    { "required": ["background_model_check"] },
                    { "required": ["cooldown"] }
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
            "rate_limit": { "$ref": "#/$defs/rate_limit" },
            "team_id": { "type": "string", "minLength": 1 },
            "owner_id": { "type": "string", "minLength": 1 }
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
    // `provider`, `adapter`, and `telemetry_tags` were added as a
    // skeleton for issue #302 Phase A (PR #298). `request` and
    // `response` were added in Phase A2.5 to land the on-disk shape
    // for the `RuntimeConfig.request` / `RuntimeConfig.response`
    // blocks from issue #302 §5. All Phase A fields are optional on
    // the wire (matching `#[serde(default)]` on the Rust side) so
    // existing ProviderKey payloads without these fields keep
    // validating. No dispatch path reads them in this PR.
    json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["display_name", "secret"],
        "additionalProperties": false,
        "properties": {
            "display_name": { "type": "string", "minLength": 1 },
            "secret":       { "type": "string", "minLength": 1 },
            "api_base":     { "type": "string" },
            // Phase A skeleton — vendor identity, free-form string.
            // Closed-set validation is deferred to a follow-up Phase A
            // PR that wires dispatch onto `provider`.
            "provider":     { "type": "string" },
            // Phase A skeleton — wire-shape adapter. Pinned to the
            // closed Adapter enum.
            "adapter":      { "type": "string", "enum": ["openai", "anthropic", "bedrock", "vertex", "azure-openai"] },
            "telemetry_tags": {
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "kind":             { "type": "string", "enum": ["catalog", "byo"] },
                    "featured":         { "type": "boolean" },
                    "branded_provider": { "type": ["string", "null"] },
                    "pk_label":         { "type": ["string", "null"] },
                    "byo_label":        { "type": ["string", "null"] }
                }
            },
            // Phase A2.5 — RuntimeConfig.request, see issue #302 §5.
            // Each sub-field is the input to a primitive apply
            // function in aisix-provider-openai's overrides module.
            "request": {
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "param_renames": {
                        "type": "object",
                        "additionalProperties": { "type": "string" }
                    },
                    "param_constraints": {
                        "type": "object",
                        "additionalProperties": false,
                        "properties": {
                            "temperature_max": { "type": "number" },
                            "temperature_min": { "type": "number" }
                        }
                    },
                    "default_headers": {
                        "type": "object",
                        "additionalProperties": { "type": "string" }
                    },
                    // Free-form on purpose — the cp-api spec lets
                    // operators set any default top-level body field
                    // (`safe_prompt`, `transforms`, etc.); the apply
                    // path only adds keys when the caller did not
                    // set them.
                    "default_body_fields": {
                        "type": "object"
                    }
                }
            },
            // Phase A2.5 — RuntimeConfig.response, see issue #302 §5.
            "response": {
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "stream_done_marker":     { "type": "string", "enum": ["required", "optional", "none"] },
                    "content_list_to_string": { "type": "boolean" },
                    // Open string in Phase A2.5 — matches the Rust
                    // `Option<String>`. Phase D pins the closed
                    // ("openai" | "passthrough") set.
                    "error_envelope":         { "type": "string" },
                    "reasoning_field":        { "type": "string" }
                }
            },
            // Issue #411 — per-PK passthrough header strip list.
            // Optional (defaults applied DP-side via
            // `#[serde(default = "default_strip_headers")]`); when
            // present, must be an array of strings. Entries are
            // normalised (trim/lowercase/dedup/drop-empties) on
            // deserialize so this validator doesn't enforce
            // formatting beyond the type shape.
            "strip_headers": {
                "type": "array",
                "items": { "type": "string" }
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

fn rate_limit_policy_schema() -> Value {
    json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["name", "scope", "scope_ref", "window"],
        "additionalProperties": false,
        "properties": {
            "name":         { "type": "string", "minLength": 1 },
            "scope":        { "type": "string", "enum": ["api_key", "model", "team", "member"] },
            "scope_ref":    { "type": "string", "minLength": 1 },
            "window":       { "type": "string", "enum": ["second", "minute", "hour"] },
            "max_requests": { "type": "integer", "minimum": 1 },
            "max_tokens":   { "type": "integer", "minimum": 1 }
        },
        "anyOf": [
            { "required": ["max_requests"] },
            { "required": ["max_tokens"] }
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
    fn apikey_with_team_and_owner_fields_passes() {
        let v = json!({
            "key_hash":"9df37f5e7cbc3c391d872742b5f286c242e733a09add9eeaa4d26a599bd90b20",
            "allowed_models":["gpt-4o"],
            "team_id": "team-uuid-1",
            "owner_id": "member-uuid-1"
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
    fn routing_on_all_filtered_fail_passes() {
        let v = json!({
            "display_name": "router-1",
            "routing": {
                "targets": [{"model": "a"}],
                "on_all_filtered": "fail"
            }
        });
        validate_model(&v).unwrap();
    }

    #[test]
    fn routing_on_all_filtered_original_order_passes() {
        let v = json!({
            "display_name": "router-1",
            "routing": {
                "targets": [{"model": "a"}],
                "on_all_filtered": "original_order"
            }
        });
        validate_model(&v).unwrap();
    }

    #[test]
    fn routing_on_all_filtered_rejects_unknown_value() {
        let v = json!({
            "display_name": "router-1",
            "routing": {
                "targets": [{"model": "a"}],
                "on_all_filtered": "yolo"
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
}
