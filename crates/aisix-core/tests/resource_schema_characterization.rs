//! Characterization (golden-corpus) tests for the resource validators that
//! were migrated from hand-written `json!` schemas to struct-derived schemas
//! (single source of truth). Each corpus pins the exact accept/reject behavior
//! so the migration can prove it preserved the config contract (except the
//! documented intended changes — e.g. rate_limit `rps`/`rph` on api_key).
//!
//! One table per resource; the label is printed on failure so the offending
//! case is obvious. New resources append their own table as they migrate.

use aisix_core::models::schema::{
    resource_root_schema, unknown_field_paths, validate_apikey, validate_cache_policy,
    validate_guardrail, validate_guardrail_attachment, validate_observability_exporter,
    validate_provider_key, validate_rate_limit_policy, RESOURCES,
};
use serde_json::{json, Value};
use std::fs;
use std::path::Path;

/// Run a corpus of `(label, expect_accept, payload)` against `validate`.
///
/// Every accepted case doubles as a guard on the loader's unknown-field
/// report: a document the write contract accepts is a document this build
/// fully understands, so it must report nothing. The report is derived from
/// the strict schema (`unknown_field_paths`), and it runs on every row of
/// these kinds — a producer that stops declaring a field would turn the whole
/// corpus into permanent partial-compat warnings.
#[track_caller]
fn check(
    resource: &str,
    validate: fn(&Value) -> Result<(), aisix_core::models::schema::SchemaError>,
    cases: &[(&str, bool, Value)],
) {
    for (label, expect_accept, payload) in cases {
        let result = validate(payload);
        if *expect_accept {
            assert!(
                result.is_ok(),
                "expected ACCEPT for `{label}`, got: {:?}",
                result.err()
            );
            let unknown = unknown_field_paths(resource, payload);
            assert!(
                unknown.is_empty(),
                "`{label}` is fully valid but reports unknown fields: {unknown:?}"
            );
        } else {
            assert!(
                result.is_err(),
                "expected REJECT for `{label}`, but it was accepted"
            );
        }
    }
}

#[test]
fn cache_policy_corpus() {
    check(
        "cache_policy",
        validate_cache_policy,
        &[
            (
                "minimal (only required name)",
                true,
                json!({"name": "prod-default"}),
            ),
            (
                "full redis policy",
                true,
                json!({"name": "shared", "enabled": false, "backend": "redis", "ttl_seconds": 600, "applies_to": "model:gpt-4o"}),
            ),
            (
                "ttl_seconds at lower bound",
                true,
                json!({"name": "x", "ttl_seconds": 1}),
            ),
            (
                "ttl_seconds at upper bound",
                true,
                json!({"name": "x", "ttl_seconds": 604800}),
            ),
            (
                "applies_to api_key scope",
                true,
                json!({"name": "k", "applies_to": "api_key:11111111-1111-1111-1111-111111111111"}),
            ),
            (
                "model scope by resource id",
                true,
                json!({"name": "k", "applies_to_model_id": "m-1"}),
            ),
            (
                "similarity embedder named by resource id alone",
                true,
                json!({"name": "k", "semantic": {"embedding_model_id": "m-e", "threshold": 0.9}}),
            ),
            (
                "similarity embedder named neither way",
                false,
                json!({"name": "k", "semantic": {"threshold": 0.9}}),
            ),
            (
                "empty applies_to_model_id",
                false,
                json!({"name": "k", "applies_to_model_id": ""}),
            ),
            // CachePolicy has no deny_unknown_fields → forward-compat fields tolerated.
            (
                "unknown field tolerated",
                true,
                json!({"name": "future", "backend": "memory", "future_knob": "ignored"}),
            ),
            ("missing required name", false, json!({"backend": "memory"})),
            ("empty name", false, json!({"name": ""})),
            (
                "name over 120 chars",
                false,
                json!({"name": "a".repeat(121)}),
            ),
            (
                "ttl_seconds below minimum (0)",
                false,
                json!({"name": "x", "ttl_seconds": 0}),
            ),
            (
                "ttl_seconds above maximum",
                false,
                json!({"name": "x", "ttl_seconds": 604801}),
            ),
            (
                "unknown backend enum",
                false,
                json!({"name": "x", "backend": "semantic"}),
            ),
            (
                "empty applies_to",
                false,
                json!({"name": "x", "applies_to": ""}),
            ),
            (
                "applies_to over 255 chars",
                false,
                json!({"name": "x", "applies_to": "m".repeat(256)}),
            ),
        ],
    );
}

#[test]
fn apikey_corpus() {
    check(
        "api_key",
        validate_apikey,
        &[
            (
                "happy path",
                true,
                json!({"key_hash": "h", "allowed_models": ["a", "b"]}),
            ),
            // Empty allowed_models is a deny-all (runtime semantics), valid shape.
            (
                "empty allowed_models",
                true,
                json!({"key_hash": "h", "allowed_models": []}),
            ),
            // Neither grant field: a key may grant models by name
            // (`allowed_models`) or by resource id (`allowed_model_ids`),
            // so neither is required — carrying neither is a valid
            // document that grants no model access.
            ("no grant field at all", true, json!({"key_hash": "h"})),
            (
                "allowed_model_ids alone",
                true,
                json!({"key_hash": "h", "allowed_model_ids": ["m-1"]}),
            ),
            (
                "allowed_model_ids of non-strings",
                false,
                json!({"key_hash": "h", "allowed_model_ids": [1]}),
            ),
            ("missing key_hash", false, json!({"allowed_models": ["a"]})),
            (
                "empty key_hash",
                false,
                json!({"key_hash": "", "allowed_models": ["a"]}),
            ),
            (
                "unknown top-level field",
                false,
                json!({"key_hash": "h", "allowed_models": ["a"], "bogus": 1}),
            ),
            (
                "rate_limit ok",
                true,
                json!({"key_hash": "h", "allowed_models": ["a"], "rate_limit": {"rpm": 60, "concurrency": 5}}),
            ),
            (
                "rate_limit unknown dim",
                false,
                json!({"key_hash": "h", "allowed_models": ["a"], "rate_limit": {"bogus": 1}}),
            ),
            (
                "string team/user",
                true,
                json!({"key_hash": "h", "allowed_models": ["a"], "team_id": "t1", "user_id": "m1"}),
            ),
            // The load-bearing nullable case: cp-api sends null to clear team/owner.
            (
                "null team and user",
                true,
                json!({"key_hash": "h", "allowed_models": ["a"], "team_id": null, "user_id": null}),
            ),
            (
                "one null one absent",
                true,
                json!({"key_hash": "h", "allowed_models": ["a"], "team_id": null}),
            ),
            (
                "null rate_limit",
                true,
                json!({"key_hash": "h", "allowed_models": ["a"], "rate_limit": null}),
            ),
            (
                "empty team_id",
                false,
                json!({"key_hash": "h", "allowed_models": ["a"], "team_id": ""}),
            ),
            (
                "non-string allowed_models item",
                false,
                json!({"key_hash": "h", "allowed_models": [1, 2]}),
            ),
            (
                "negative rate_limit dim",
                false,
                json!({"key_hash": "h", "allowed_models": ["a"], "rate_limit": {"rpm": -1}}),
            ),
            // The shared-RateLimit rps/rph fix (also applied to api_key).
            (
                "rate_limit rps/rph accepted",
                true,
                json!({"key_hash": "h", "allowed_models": ["a"], "rate_limit": {"rps": 5, "rph": 100}}),
            ),
        ],
    );
}

#[test]
fn rate_limit_policy_corpus() {
    check(
        "rate_limit_policy",
        validate_rate_limit_policy,
        &[
            (
                "full",
                true,
                json!({"name": "q", "scope": "team", "scope_ref": "t1", "window": "minute", "max_requests": 100, "max_tokens": 50000}),
            ),
            (
                "only max_requests (anyOf)",
                true,
                json!({"name": "q", "scope": "api_key", "scope_ref": "k1", "window": "minute", "max_requests": 60}),
            ),
            (
                "only max_tokens (anyOf)",
                true,
                json!({"name": "q", "scope": "member", "scope_ref": "m1", "window": "hour", "max_tokens": 1000000}),
            ),
            (
                "team_member + second window",
                true,
                json!({"name": "q", "scope": "team_member", "scope_ref": "t1", "window": "second", "max_requests": 10}),
            ),
            (
                "neither cap present (anyOf)",
                false,
                json!({"name": "q", "scope": "team", "scope_ref": "t1", "window": "minute"}),
            ),
            (
                "missing name",
                false,
                json!({"scope": "team", "scope_ref": "t1", "window": "minute", "max_requests": 1}),
            ),
            (
                "unknown scope enum",
                false,
                json!({"name": "q", "scope": "region", "scope_ref": "t1", "window": "minute", "max_requests": 1}),
            ),
            (
                "team + day window token quota (#771)",
                true,
                json!({"name": "q", "scope": "team", "scope_ref": "t1", "window": "day", "max_tokens": 100}),
            ),
            (
                "unknown window enum",
                false,
                json!({"name": "q", "scope": "team", "scope_ref": "t1", "window": "week", "max_requests": 1}),
            ),
            (
                "max_requests below minimum (0)",
                false,
                json!({"name": "q", "scope": "team", "scope_ref": "t1", "window": "minute", "max_requests": 0}),
            ),
            (
                "max_tokens below minimum (0)",
                false,
                json!({"name": "q", "scope": "team", "scope_ref": "t1", "window": "minute", "max_tokens": 0}),
            ),
            (
                "empty name",
                false,
                json!({"name": "", "scope": "team", "scope_ref": "t1", "window": "minute", "max_requests": 1}),
            ),
            (
                "empty scope_ref",
                false,
                json!({"name": "q", "scope": "team", "scope_ref": "", "window": "minute", "max_requests": 1}),
            ),
            (
                "unknown field",
                false,
                json!({"name": "q", "scope": "team", "scope_ref": "t1", "window": "minute", "max_requests": 1, "extra": true}),
            ),
            (
                "negative max_requests",
                false,
                json!({"name": "q", "scope": "team", "scope_ref": "t1", "window": "minute", "max_requests": -1}),
            ),
        ],
    );
}

#[test]
fn provider_key_corpus() {
    // Corpus fixtures deliberately keep the credential's former `secret`
    // spelling — they double as acceptance proof for stored documents
    // written before the field's rename to `api_key`. The canonical
    // spelling is pinned by the dedicated cases below.
    check(
        "provider_key",
        validate_provider_key,
        &[
            (
                "minimal (former `secret` spelling)",
                true,
                json!({"display_name": "openai-prod", "secret": "sk-x"}),
            ),
            (
                "minimal (canonical `api_key` spelling)",
                true,
                json!({"display_name": "openai-prod", "api_key": "sk-x"}),
            ),
            (
                "with api_base + provider",
                true,
                json!({"display_name": "p", "secret": "sk-x", "api_base": "https://api.openai.com/v1", "provider": "deepseek"}),
            ),
            ("missing display_name", false, json!({"secret": "sk-x"})),
            (
                "credential absent under both spellings",
                false,
                json!({"display_name": "x"}),
            ),
            (
                "both credential spellings (schema layer admits; serde rejects the duplicate)",
                true,
                json!({"display_name": "x", "api_key": "a", "secret": "b"}),
            ),
            (
                "unknown top-level field",
                false,
                json!({"display_name": "x", "secret": "k", "rogue": 1}),
            ),
            (
                "empty display_name",
                false,
                json!({"display_name": "", "secret": "k"}),
            ),
            (
                "empty api_key",
                false,
                json!({"display_name": "x", "api_key": ""}),
            ),
            (
                "empty secret",
                false,
                json!({"display_name": "x", "secret": ""}),
            ),
            (
                "adapter azure-openai",
                true,
                json!({"display_name": "x", "secret": "k", "adapter": "azure-openai"}),
            ),
            (
                "adapter invalid",
                false,
                json!({"display_name": "x", "secret": "k", "adapter": "not-a-real-adapter"}),
            ),
            // option_add_null_type=true: optional fields accept explicit null.
            (
                "adapter null",
                true,
                json!({"display_name": "x", "secret": "k", "adapter": null}),
            ),
            (
                "telemetry catalog",
                true,
                json!({"display_name": "x", "secret": "k", "telemetry_tags": {"kind": "catalog", "featured": true, "branded_provider": "deepseek", "pk_label": "prod"}}),
            ),
            (
                "telemetry byo, branded omitted",
                true,
                json!({"display_name": "x", "secret": "k", "telemetry_tags": {"kind": "byo", "byo_label": "platform-team"}}),
            ),
            // The load-bearing nullable case: cp-api sends branded_provider:null.
            (
                "telemetry branded_provider null",
                true,
                json!({"display_name": "x", "secret": "k", "telemetry_tags": {"branded_provider": null}}),
            ),
            (
                "telemetry unknown tag",
                false,
                json!({"display_name": "x", "secret": "k", "telemetry_tags": {"unknown_tag": "v"}}),
            ),
            (
                "telemetry kind invalid (closed enum)",
                false,
                json!({"display_name": "x", "secret": "k", "telemetry_tags": {"kind": "third-party"}}),
            ),
            (
                "request empty",
                true,
                json!({"display_name": "x", "secret": "k", "request": {}}),
            ),
            (
                "request full",
                true,
                json!({"display_name": "x", "secret": "k", "request": {"param_renames": {"max_completion_tokens": "max_tokens"}, "param_constraints": {"temperature_max": 1.0}, "default_headers": {"X-Foo": "bar"}, "default_body_fields": {"safe_prompt": true}}}),
            ),
            (
                "request typo field",
                false,
                json!({"display_name": "x", "secret": "k", "request": {"param_rename": {}}}),
            ),
            (
                "param_constraints unknown field",
                false,
                json!({"display_name": "x", "secret": "k", "request": {"param_constraints": {"top_p_max": 0.9}}}),
            ),
            (
                "response full",
                true,
                json!({"display_name": "x", "secret": "k", "response": {"stream_done_marker": "none", "content_list_to_string": false, "error_envelope": "openai", "reasoning_field": "delta.reasoning_content"}}),
            ),
            (
                "response bad stream_done_marker",
                false,
                json!({"display_name": "x", "secret": "k", "response": {"stream_done_marker": "maybe"}}),
            ),
            (
                "response stream_done_marker case-sensitive",
                false,
                json!({"display_name": "x", "secret": "k", "response": {"stream_done_marker": "Required"}}),
            ),
            (
                "response typo field",
                false,
                json!({"display_name": "x", "secret": "k", "response": {"reasoning_fields": "x"}}),
            ),
            (
                "strip_headers empty",
                true,
                json!({"display_name": "x", "secret": "k", "strip_headers": []}),
            ),
            (
                "strip_headers non-string item",
                false,
                json!({"display_name": "x", "secret": "k", "strip_headers": [1, 2]}),
            ),
        ],
    );
}

#[test]
fn observability_exporter_corpus() {
    check(
        "observability_exporter",
        validate_observability_exporter,
        &[
            // otlp_http
            (
                "otlp minimal",
                true,
                json!({"name": "hc", "kind": "otlp_http", "endpoint": "https://api.honeycomb.io/v1/traces"}),
            ),
            (
                "otlp loopback http",
                true,
                json!({"name": "e2e", "kind": "otlp_http", "endpoint": "http://mock-otlp:4318/v1/traces"}),
            ),
            (
                "otlp plain http non-loopback (pattern)",
                false,
                json!({"name": "x", "kind": "otlp_http", "endpoint": "http://api.honeycomb.io/v1/traces"}),
            ),
            (
                "otlp sample_rate > 1",
                false,
                json!({"name": "x", "kind": "otlp_http", "endpoint": "https://x", "sample_rate": 1.1}),
            ),
            (
                "otlp missing endpoint",
                false,
                json!({"name": "x", "kind": "otlp_http"}),
            ),
            (
                "otlp content_mode unknown",
                false,
                json!({"name": "x", "kind": "otlp_http", "endpoint": "https://x", "content_mode": "verbose"}),
            ),
            (
                "otlp content_max_bytes 0",
                false,
                json!({"name": "x", "kind": "otlp_http", "endpoint": "https://x", "content_max_bytes": 0}),
            ),
            (
                "otlp content_max_bytes > 1MiB (cap preserved)",
                false,
                json!({"name": "x", "kind": "otlp_http", "endpoint": "https://x", "content_max_bytes": 2000000}),
            ),
            // aliyun_sls
            (
                "sls full",
                true,
                json!({"name": "sls", "kind": "aliyun_sls", "endpoint": "ap-southeast-3.log.aliyuncs.com", "project": "p", "logstore": "l", "credential_ref": "r"}),
            ),
            (
                "sls missing logstore",
                false,
                json!({"name": "x", "kind": "aliyun_sls", "endpoint": "ap-southeast-3.log.aliyuncs.com", "project": "p", "credential_ref": "r"}),
            ),
            (
                "sls bad endpoint host (pattern)",
                false,
                json!({"name": "x", "kind": "aliyun_sls", "endpoint": "https://evil.example.com", "project": "p", "logstore": "l", "credential_ref": "r"}),
            ),
            (
                "sls plaintext secret (additionalProperties:false)",
                false,
                json!({"name": "x", "kind": "aliyun_sls", "endpoint": "ap-southeast-3.log.aliyuncs.com", "project": "p", "logstore": "l", "credential_ref": "r", "access_key_secret": "AKIA"}),
            ),
            // object_store
            (
                "s3 credential_ref mode",
                true,
                json!({"name": "s3", "kind": "object_store", "provider": "s3", "bucket": "b", "prefix": "p", "credential_ref": "r"}),
            ),
            (
                "s3 cloud_identity (no credential_ref)",
                true,
                json!({"name": "x", "kind": "object_store", "provider": "s3", "bucket": "b", "prefix": "p", "auth_mode": "cloud_identity"}),
            ),
            (
                "azure_blob + cloud_identity (cross-field)",
                false,
                json!({"name": "x", "kind": "object_store", "provider": "azure_blob", "bucket": "c", "prefix": "p", "auth_mode": "cloud_identity"}),
            ),
            (
                "credential_ref mode missing credential_ref (else)",
                false,
                json!({"name": "x", "kind": "object_store", "provider": "s3", "bucket": "b", "prefix": "p"}),
            ),
            (
                "bad provider enum",
                false,
                json!({"name": "x", "kind": "object_store", "provider": "wasabi", "bucket": "b", "prefix": "p", "credential_ref": "r"}),
            ),
            (
                "loopback minio endpoint",
                true,
                json!({"name": "x", "kind": "object_store", "provider": "s3", "bucket": "b", "prefix": "p", "endpoint": "http://minio:9000", "credential_ref": "r"}),
            ),
            (
                "object_store empty credential_ref",
                false,
                json!({"name": "x", "kind": "object_store", "provider": "s3", "bucket": "b", "prefix": "p", "credential_ref": ""}),
            ),
            // datadog
            (
                "datadog allow-list site",
                true,
                json!({"name": "dd", "kind": "datadog", "site": "datadoghq.eu", "credential_ref": "r", "service": "s"}),
            ),
            (
                "datadog non-allow-list site (pattern)",
                false,
                json!({"name": "x", "kind": "datadog", "site": "datadoghq.org", "credential_ref": "r", "service": "s"}),
            ),
            (
                "datadog content_max_bytes > 1MiB",
                false,
                json!({"name": "x", "kind": "datadog", "site": "datadoghq.com", "credential_ref": "r", "service": "s", "content_max_bytes": 1048577}),
            ),
            // Cross-kind field leakage now rejected (per-branch additionalProperties:false).
            (
                "datadog carrying otlp/sls field",
                false,
                json!({"name": "x", "kind": "datadog", "site": "datadoghq.com", "credential_ref": "r", "service": "s", "project": "leaked"}),
            ),
            // shared / discriminator
            (
                "unknown kind",
                false,
                json!({"name": "x", "kind": "splunk_hec", "endpoint": "https://x"}),
            ),
            (
                "missing name",
                false,
                json!({"kind": "otlp_http", "endpoint": "https://x"}),
            ),
            (
                "name too long (>120)",
                false,
                json!({"name": "a".repeat(121), "kind": "otlp_http", "endpoint": "https://x"}),
            ),
        ],
    );
}

#[test]
fn guardrail_corpus() {
    check(
        "guardrail",
        validate_guardrail,
        &[
            // keyword
            (
                "keyword empty patterns",
                true,
                json!({"name": "k", "kind": "keyword", "patterns": []}),
            ),
            (
                "keyword literal + regex",
                true,
                json!({"name": "k", "kind": "keyword", "patterns": [{"kind": "literal", "value": "AKIA"}, {"kind": "regex", "value": "\\d{3}"}]}),
            ),
            (
                "keyword missing patterns",
                false,
                json!({"name": "k", "kind": "keyword"}),
            ),
            (
                "empty name",
                false,
                json!({"name": "", "kind": "keyword", "patterns": []}),
            ),
            (
                "keyword pattern empty value",
                false,
                json!({"name": "k", "kind": "keyword", "patterns": [{"kind": "literal", "value": ""}]}),
            ),
            (
                "keyword pattern bad kind",
                false,
                json!({"name": "k", "kind": "keyword", "patterns": [{"kind": "glob", "value": "x"}]}),
            ),
            (
                "keyword pattern extra field",
                false,
                json!({"name": "k", "kind": "keyword", "patterns": [{"kind": "literal", "value": "x", "extra": 1}]}),
            ),
            (
                "semantic embedder named by resource id alone",
                true,
                json!({"name": "s", "kind": "semantic", "embedding_model_id": "m-e",
                       "deny_examples": ["x"], "deny_threshold": 0.8}),
            ),
            (
                "semantic embedder named neither way",
                false,
                json!({"name": "s", "kind": "semantic",
                       "deny_examples": ["x"], "deny_threshold": 0.8}),
            ),
            (
                "semantic embedder id present but empty",
                false,
                json!({"name": "s", "kind": "semantic", "embedding_model_id": "",
                       "deny_examples": ["x"], "deny_threshold": 0.8}),
            ),
            // top-level / kind discriminator
            (
                "missing name",
                false,
                json!({"kind": "keyword", "patterns": []}),
            ),
            (
                "unknown kind",
                false,
                json!({"name": "k", "kind": "lakera", "patterns": []}),
            ),
            ("missing kind", false, json!({"name": "k"})),
            (
                "hook_point + p0c fields",
                true,
                json!({"name": "k", "kind": "keyword", "patterns": [], "hook_point": "input", "enforcement_mode": "monitor", "created_at": "2026-01-01T00:00:00Z"}),
            ),
            // created_at is a non-null string (the runtime validator always
            // enforced this; cp-api omits it when absent, never sends null).
            (
                "created_at null",
                false,
                json!({"name": "k", "kind": "keyword", "patterns": [], "created_at": null}),
            ),
            (
                "bad hook_point",
                false,
                json!({"name": "k", "kind": "keyword", "patterns": [], "hook_point": "sideways"}),
            ),
            // bedrock
            (
                "bedrock serial",
                true,
                json!({"name": "b", "kind": "bedrock", "guardrail_id": "gid", "guardrail_version": "DRAFT", "region": "us-east-1", "aws_credentials": {"kind": "static", "access_key_id": "AKIA", "secret_access_key": "s"}, "latency_mode": {"kind": "serial"}}),
            ),
            (
                "bedrock timed",
                true,
                json!({"name": "b", "kind": "bedrock", "guardrail_id": "gid", "guardrail_version": "1", "region": "us-east-1", "aws_credentials": {"kind": "static", "access_key_id": "AKIA", "secret_access_key": "s"}, "latency_mode": {"kind": "timed", "timeout_ms": 500}}),
            ),
            (
                "bedrock missing guardrail_id",
                false,
                json!({"name": "b", "kind": "bedrock", "guardrail_version": "1", "region": "us-east-1", "aws_credentials": {"kind": "static", "access_key_id": "a", "secret_access_key": "s"}, "latency_mode": {"kind": "serial"}}),
            ),
            (
                "bedrock timed timeout < 100",
                false,
                json!({"name": "b", "kind": "bedrock", "guardrail_id": "g", "guardrail_version": "1", "region": "us-east-1", "aws_credentials": {"kind": "static", "access_key_id": "a", "secret_access_key": "s"}, "latency_mode": {"kind": "timed", "timeout_ms": 50}}),
            ),
            (
                "bedrock latency_mode extra field",
                false,
                json!({"name": "b", "kind": "bedrock", "guardrail_id": "g", "guardrail_version": "1", "region": "us-east-1", "aws_credentials": {"kind": "static", "access_key_id": "a", "secret_access_key": "s"}, "latency_mode": {"kind": "timed", "timeout_ms": 500, "extra": 1}}),
            ),
            (
                "bedrock aws_credentials extra field",
                false,
                json!({"name": "b", "kind": "bedrock", "guardrail_id": "g", "guardrail_version": "1", "region": "us-east-1", "aws_credentials": {"kind": "static", "access_key_id": "a", "secret_access_key": "s", "junk": 1}, "latency_mode": {"kind": "serial"}}),
            ),
            // azure_content_safety
            (
                "azure cs minimal",
                true,
                json!({"name": "a", "kind": "azure_content_safety", "endpoint": "https://x.cognitiveservices.azure.com", "api_key": "k"}),
            ),
            (
                "azure cs missing endpoint",
                false,
                json!({"name": "a", "kind": "azure_content_safety", "api_key": "k"}),
            ),
            (
                "azure cs timeout overflow (u32)",
                false,
                json!({"name": "a", "kind": "azure_content_safety", "endpoint": "https://x", "api_key": "k", "timeout_ms": 4_294_967_296u64}),
            ),
            // azure_content_safety_text_moderation
            (
                "azure tm minimal",
                true,
                json!({"name": "m", "kind": "azure_content_safety_text_moderation", "endpoint": "https://x", "api_key": "k"}),
            ),
            (
                "azure tm full",
                true,
                json!({"name": "m", "kind": "azure_content_safety_text_moderation", "endpoint": "https://x", "api_key": "k", "output_type": "EightSeverityLevels", "categories": ["Hate", "Violence"], "severity_threshold": 0, "stream_processing_mode": "buffer_full", "window_size": 5000, "on_buffer_exceeded": "fail_open"}),
            ),
            (
                "azure tm severity > 7",
                false,
                json!({"name": "m", "kind": "azure_content_safety_text_moderation", "endpoint": "https://x", "api_key": "k", "severity_threshold": 8}),
            ),
            (
                "azure tm window_size > 10000",
                false,
                json!({"name": "m", "kind": "azure_content_safety_text_moderation", "endpoint": "https://x", "api_key": "k", "window_size": 20000}),
            ),
            (
                "azure tm output_type enum (injected)",
                false,
                json!({"name": "m", "kind": "azure_content_safety_text_moderation", "endpoint": "https://x", "api_key": "k", "output_type": "Twelve"}),
            ),
            (
                "azure tm categories item enum (injected)",
                false,
                json!({"name": "m", "kind": "azure_content_safety_text_moderation", "endpoint": "https://x", "api_key": "k", "categories": ["Nope"]}),
            ),
            // aliyun_text_moderation
            (
                "aliyun minimal",
                true,
                json!({"name": "al", "kind": "aliyun_text_moderation", "region": "cn-shanghai", "access_key_id": "LTAI", "access_key_secret": "s"}),
            ),
            (
                "aliyun missing region",
                false,
                json!({"name": "al", "kind": "aliyun_text_moderation", "access_key_id": "id", "access_key_secret": "s"}),
            ),
            (
                "aliyun risk_level enum (injected)",
                false,
                json!({"name": "al", "kind": "aliyun_text_moderation", "region": "cn", "access_key_id": "id", "access_key_secret": "s", "risk_level_threshold": "critical"}),
            ),
            (
                "aliyun window_size > 2000",
                false,
                json!({"name": "al", "kind": "aliyun_text_moderation", "region": "cn", "access_key_id": "id", "access_key_secret": "s", "window_size": 3000}),
            ),
        ],
    );
}

#[test]
fn guardrail_attachment_corpus() {
    check(
        "guardrail_attachment",
        validate_guardrail_attachment,
        &[
            (
                "env scope null scope_id",
                true,
                json!({"guardrail_id": "gid", "scope_type": "env", "scope_id": null, "priority": 0}),
            ),
            (
                "model scope",
                true,
                json!({"guardrail_id": "gid", "scope_type": "model", "scope_id": "mid", "priority": 10, "enabled": false}),
            ),
            // Non-`env` scope with null/absent scope_id is accepted — the
            // original validator never conditionally required scope_id, and the
            // runtime resolver tolerates None. Pinned to keep that contract.
            (
                "model scope null scope_id",
                true,
                json!({"guardrail_id": "gid", "scope_type": "model", "scope_id": null, "priority": 1}),
            ),
            (
                "team scope negative priority",
                true,
                json!({"guardrail_id": "gid", "scope_type": "team", "scope_id": "tid", "priority": -5}),
            ),
            (
                "api_key scope_id omitted",
                true,
                json!({"guardrail_id": "gid", "scope_type": "api_key", "priority": 1}),
            ),
            (
                "extra field tolerated (open)",
                true,
                json!({"guardrail_id": "gid", "scope_type": "env", "priority": 1, "env_id": "e1"}),
            ),
            (
                "missing guardrail_id",
                false,
                json!({"scope_type": "env", "priority": 1}),
            ),
            (
                "empty guardrail_id",
                false,
                json!({"guardrail_id": "", "scope_type": "env", "priority": 1}),
            ),
            (
                "bad scope_type enum",
                false,
                json!({"guardrail_id": "gid", "scope_type": "org", "priority": 1}),
            ),
            (
                "missing scope_type",
                false,
                json!({"guardrail_id": "gid", "priority": 1}),
            ),
            (
                "missing priority",
                false,
                json!({"guardrail_id": "gid", "scope_type": "env"}),
            ),
            (
                "priority not integer",
                false,
                json!({"guardrail_id": "gid", "scope_type": "env", "priority": "high"}),
            ),
        ],
    );
}

// ---------------------------------------------------------------------------
// Published schema files — `schemas/resources/` (strict, the write contract)
// and `schemas/resources-lenient/` (the etcd loader's read contract).
//
// The corpora above pin what the in-process validators do. This section pins
// that the files a downstream consumer vendors ARE those validators, so the
// control plane can read the read contract instead of re-deriving it from the
// strict files.
// ---------------------------------------------------------------------------

/// Parse one published schema file out of `schemas/<dir>/`.
fn schemas_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("CARGO_MANIFEST_DIR has two ancestors")
        .join("schemas")
}

/// Sorted file names published under `schemas/<sub>/`. Read off the directory
/// rather than a hard-coded list so a file the dump starts (or stops) emitting
/// reaches the checks below.
fn published_file_names(sub: &str) -> Vec<String> {
    let mut names: Vec<String> = fs::read_dir(schemas_dir().join(sub))
        .unwrap_or_else(|e| panic!("read schemas/{sub}: {e}"))
        .map(|e| {
            e.expect("dir entry")
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    names.sort();
    names
}

fn published_schema(dir: &str, resource: &str) -> Value {
    let path = schemas_dir()
        .join(dir)
        .join(format!("{resource}.schema.json"));
    let bytes =
        fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&bytes).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

/// Every published lenient file is exactly what `LENIENT_SCHEMAS` compiles.
///
/// `Schemas::compile(false)` builds each validator from
/// `resource_root_schema(resource, false)`, so equality here is the whole
/// provenance claim: the file is the loader's schema, not a transformation of
/// the strict one that happens to agree today.
#[test]
fn published_lenient_schemas_are_what_the_loader_compiles() {
    for resource in RESOURCES {
        assert_eq!(
            published_schema("resources-lenient", resource),
            resource_root_schema(resource, false),
            "schemas/resources-lenient/{resource}.schema.json is not \
             resource_root_schema({resource:?}, false) — re-run \
             `cargo run -p aisix-core --bin dump-schema`"
        );
    }
}

/// The strict twin of the check above: every published strict file is exactly
/// what the write validators compile. Those files are also `include_str!`ed
/// into the DP admin OpenAPI document, so they are worth pinning here and not
/// only in the CI drift job.
#[test]
fn published_strict_schemas_are_what_the_write_path_compiles() {
    for resource in RESOURCES {
        assert_eq!(
            published_schema("resources", resource),
            resource_root_schema(resource, true),
            "schemas/resources/{resource}.schema.json is not \
             resource_root_schema({resource:?}, true) — re-run \
             `cargo run -p aisix-core --bin dump-schema`"
        );
    }
}

/// Where the two published sets differ BEYOND unknown fields, pinned as the
/// exhaustive list of JSON paths at which the strict file (with every
/// `additionalProperties: false` stripped) and the lenient file disagree.
///
/// A consumer that models the lenient set as "the strict set with the
/// closures removed" is wrong for these four, and the difference is not
/// cosmetic: the loader accepts an `mcp_policy` with no `allow` and reads the
/// field's default. Registering the paths rather than a prose reason is what
/// makes the table checkable — a claim that some OTHER field relaxed, or that
/// one of these stopped relaxing, moves a path and fails.
///
/// Two shapes appear here, and `schemas/README.md` must keep telling them
/// apart:
///
/// - a `required` / `not` change, which really does let the loader accept a
///   document the write path rejects (`api_key`, `mcp_policy`, `model`, and
///   the `semantic` guardrail branch);
/// - a `default` annotation the STRICT producer strips on purpose and the
///   lenient one keeps, which changes nothing about what validates but does
///   feed a schema-driven form generator a value the same branch would refuse
///   (the `custom` guardrail's `script`, whose `default: ""` sits beside
///   `minLength: 1`; the semantic thresholds' `default: 0.75`). `script`
///   itself is required on BOTH sets.
const EXTRA_RELAXATIONS: &[(&str, &[&str])] = &[
    ("api_key", &["/definitions/McpAccess/required"]),
    (
        "guardrail",
        &[
            "/oneOf/10/allOf",
            "/oneOf/10/properties/allow_threshold/default",
            "/oneOf/10/properties/deny_threshold/default",
            "/oneOf/11/properties/script/default",
        ],
    ),
    ("mcp_policy", &["/required"]),
    (
        "model",
        &[
            "/oneOf/0/not/anyOf",
            "/oneOf/1/not/anyOf",
            "/oneOf/2/not/anyOf",
            "/oneOf/3/not/anyOf",
        ],
    ),
];

/// Every field the resources file refuses as an id-form model reference
/// is a field this build's schema actually declares.
///
/// The refusal list (`filesource::model_ref_id_fields`) is written by
/// hand, and `aisix export` rewrites the same list back to name form. A
/// typo in either half is silent in both directions: the file would
/// accept an id that resolves to nothing, and the export would leave one
/// in a file that then refuses to load. Neither shows up as a test
/// failure anywhere else, because a name nothing declares simply never
/// matches.
#[test]
fn every_refused_model_reference_id_is_a_declared_field() {
    fn declares(node: &Value, field: &str) -> bool {
        match node {
            Value::Object(map) => {
                map.get("properties")
                    .and_then(Value::as_object)
                    .is_some_and(|p| p.contains_key(field))
                    || map.values().any(|v| declares(v, field))
            }
            Value::Array(items) => items.iter().any(|v| declares(v, field)),
            _ => false,
        }
    }

    // (resources-file collection, the resource whose schema declares it)
    for (kind, resource) in [
        ("api_keys", "api_key"),
        ("models", "model"),
        ("cache_policies", "cache_policy"),
        ("guardrails", "guardrail"),
    ] {
        let schema = resource_root_schema(resource, true);
        let fields = aisix_core::filesource::model_ref_id_fields(kind);
        assert!(
            !fields.is_empty(),
            "{kind} has model references but refuses none"
        );
        for (id_field, name_field) in fields {
            assert!(
                declares(&schema, id_field),
                "{kind} refuses `{id_field}`, which the {resource} schema does not declare"
            );
            assert!(
                declares(&schema, name_field),
                "{kind} rewrites `{id_field}` to `{name_field}`, which the {resource} schema \
                 does not declare"
            );
        }
    }
}

/// The published files are exactly the ones `dump-schema` emits today.
///
/// `dump-schema` only ever writes, so a file it STOPPED emitting would sit in
/// the tree, keep matching its twin, and go on being vendored as a contract
/// this build no longer has. Every other check here is driven off `RESOURCES`
/// or off the directory itself, and neither can see such an orphan.
#[test]
fn published_directories_hold_exactly_what_the_dump_emits() {
    // The nested struct types `dump-schema` publishes beside the resources.
    // They have no runtime validator, so `RESOURCES` does not name them.
    const NESTED: [&str; 5] = ["embedding", "ensemble", "rate_limit", "routing", "semantic"];

    let mut expected: Vec<String> = RESOURCES
        .iter()
        .chain(NESTED.iter())
        .map(|n| format!("{n}.schema.json"))
        .collect();
    expected.sort();
    for dir in ["resources", "resources-lenient"] {
        assert_eq!(
            published_file_names(dir),
            expected,
            "schemas/{dir}/ holds a file dump-schema no longer emits, or is \
             missing one it does"
        );
    }
}

/// The two published sets differ ONLY by `additionalProperties: false`,
/// except at the paths [`EXTRA_RELAXATIONS`] registers.
///
/// Strips every closure out of the strict file and requires the result to
/// equal the lenient one. This is the claim `schemas/README.md` makes to
/// downstream consumers, and the claim the control plane cannot make
/// unconditionally when it derives one set from the other.
#[test]
fn published_sets_differ_only_where_registered() {
    fn without_closures(node: &Value) -> Value {
        match node {
            Value::Object(obj) => Value::Object(
                obj.iter()
                    .filter(|(k, v)| !(k.as_str() == "additionalProperties" && *v == &json!(false)))
                    .map(|(k, v)| (k.clone(), without_closures(v)))
                    .collect(),
            ),
            Value::Array(items) => Value::Array(items.iter().map(without_closures).collect()),
            other => other.clone(),
        }
    }

    /// Every JSON path at which `a` and `b` disagree, deepest name that still
    /// differs. A length mismatch reports the array itself.
    fn diff_paths(a: &Value, b: &Value, at: &str, out: &mut Vec<String>) {
        match (a, b) {
            (Value::Object(x), Value::Object(y)) => {
                let mut keys: Vec<&String> = x.keys().chain(y.keys()).collect();
                keys.sort();
                keys.dedup();
                for k in keys {
                    match (x.get(k), y.get(k)) {
                        (Some(l), Some(r)) => diff_paths(l, r, &format!("{at}/{k}"), out),
                        _ => out.push(format!("{at}/{k}")),
                    }
                }
            }
            (Value::Array(x), Value::Array(y)) if x.len() == y.len() => {
                for (i, (l, r)) in x.iter().zip(y).enumerate() {
                    diff_paths(l, r, &format!("{at}/{i}"), out);
                }
            }
            _ if a != b => out.push(at.to_string()),
            _ => {}
        }
    }

    let mut found: Vec<(String, Vec<String>)> = Vec::new();
    for name in published_file_names("resources-lenient") {
        let resource = name.trim_end_matches(".schema.json").to_string();
        let opened = without_closures(&published_schema("resources", &resource));
        let mut paths = Vec::new();
        diff_paths(
            &opened,
            &published_schema("resources-lenient", &resource),
            "",
            &mut paths,
        );
        paths.sort();
        if !paths.is_empty() {
            found.push((resource, paths));
        }
    }
    found.sort();

    let mut registered: Vec<(String, Vec<String>)> = EXTRA_RELAXATIONS
        .iter()
        .map(|(r, paths)| {
            (
                (*r).to_string(),
                paths.iter().map(|p| (*p).to_string()).collect(),
            )
        })
        .collect();
    registered.sort();
    assert_eq!(
        found, registered,
        "the published sets differ beyond `additionalProperties: false` at a \
         path EXTRA_RELAXATIONS does not register (or register one that no \
         longer differs). schemas/README.md describes this list in prose — \
         update both."
    );
}

/// No published lenient file closes anything, at any depth.
///
/// This is the property `open_unknown_fields` exists for and the one a
/// consumer of these files relies on: a closure left standing anywhere — a
/// `definitions` entry, a `oneOf` branch, a nested property — is a whole
/// stored row lost the first time a newer control plane writes a field under
/// it (#1014). Walks the directory rather than a hard-coded list so a file
/// the dump starts emitting cannot skip the check.
#[test]
fn published_lenient_schemas_close_nothing_at_any_depth() {
    fn closed_paths(node: &Value, path: &str, out: &mut Vec<String>) {
        match node {
            Value::Object(obj) => {
                if obj.get("additionalProperties") == Some(&Value::Bool(false)) {
                    out.push(path.to_string());
                }
                for (k, v) in obj {
                    closed_paths(v, &format!("{path}/{k}"), out);
                }
            }
            Value::Array(items) => {
                for (i, v) in items.iter().enumerate() {
                    closed_paths(v, &format!("{path}/{i}"), out);
                }
            }
            _ => {}
        }
    }

    // The two sets publish the same resources under the same file names — the
    // lenient set is a full twin, not a subset of interesting cases.
    assert_eq!(
        published_file_names("resources"),
        published_file_names("resources-lenient")
    );

    for name in published_file_names("resources-lenient") {
        let resource = name.trim_end_matches(".schema.json");
        let schema = published_schema("resources-lenient", resource);
        let mut closed = Vec::new();
        closed_paths(&schema, "", &mut closed);
        assert!(
            closed.is_empty(),
            "{name} still closes unknown fields at {closed:?}"
        );
    }
}

/// One probe case: a valid `resource` document, and the JSON pointer to the
/// object the unknown field is inserted into (`""` selects the root).
struct Probe {
    resource: &'static str,
    pointer: &'static str,
    document: fn() -> Value,
}

/// One valid document per resource, pointed at a position that resource's
/// write contract closes — nested wherever the resource has a nested closure.
/// The test inserts a field no build knows there: the strict file must then
/// reject the document it accepted a moment ago, and the lenient file must
/// still accept it.
///
/// Asserting the un-probed document passes the strict file is what makes the
/// rejection attributable to the unknown field rather than to anything else in
/// the fixture (a missing credential, an unsatisfied `oneOf`).
///
/// A non-empty pointer is the case a root-only check cannot see, and the one
/// an older gateway got wrong before #1014 — a field added inside a nested
/// config object took the whole row down. Six resources close only their root
/// and say so with `""` rather than silently testing the weaker property.
const UNKNOWN_FIELD_TOLERANCE: &[Probe] = &[
    Probe {
        resource: "model",
        pointer: "/rate_limit",
        document: || {
            json!({"display_name": "m", "provider": "openai", "model_name": "gpt-4o",
                   "provider_key_id": "pk-1", "rate_limit": {"rpm": 10}})
        },
    },
    Probe {
        resource: "api_key",
        pointer: "/rate_limit",
        document: || json!({"key_hash": "h", "allowed_models": ["a"], "rate_limit": {"rpm": 10}}),
    },
    Probe {
        resource: "provider_key",
        pointer: "/tls",
        document: || json!({"display_name": "pk", "api_key": "sk-x", "tls": {"verify": false}}),
    },
    Probe {
        resource: "guardrail",
        pointer: "/detectors/0",
        document: || json!({"name": "g", "kind": "pii", "detectors": [{"type": "email"}]}),
    },
    Probe {
        resource: "rate_limit_policy",
        pointer: "/limits",
        document: || json!({"name": "p", "limits": {"rpm": 10}}),
    },
    Probe {
        resource: "claim_mapping",
        pointer: "/resolve",
        document: || {
            json!({"name": "c", "jwt_provider": "p",
                   "match": [{"claim": "sub", "op": "exact", "values": ["a"]}],
                   "resolve": {"api_key_id": "k"}})
        },
    },
    Probe {
        resource: "mcp_auth_settings",
        pointer: "/anonymous",
        document: || {
            json!({"anonymous": {"api_key_id": "k", "servers": ["s"],
                                 "source_cidrs": ["10.0.0.0/8"]}})
        },
    },
    // Root-only: these resources embed no object their write contract closes.
    Probe {
        resource: "observability_exporter",
        pointer: "",
        document: || {
            json!({"name": "e", "kind": "otlp_http",
                   "endpoint": "https://collector.example/v1/traces"})
        },
    },
    Probe {
        resource: "mcp_server",
        pointer: "",
        document: || json!({"name": "s", "url": "https://example.com/mcp"}),
    },
    Probe {
        resource: "mcp_policy",
        pointer: "",
        document: || json!({"scope": "env", "allow": ["*"]}),
    },
    Probe {
        resource: "a2a_agent",
        pointer: "",
        document: || json!({"name": "ag", "url": "https://example.com/a2a"}),
    },
    Probe {
        resource: "oidc_provider",
        pointer: "",
        document: || json!({"name": "o", "issuer": "https://issuer.example", "audiences": ["a"]}),
    },
    Probe {
        resource: "passthrough_route",
        pointer: "",
        document: || {
            json!({"name": "r", "path_prefix": "/proxy",
                   "target_url": "https://upstream.example",
                   "provider_key_id": "pk-1"})
        },
    },
];

/// The two resources whose write contract closes NOTHING — not the root, not
/// a nested object — so no document can separate their strict and lenient
/// files. They are checked in the opposite direction: the strict file must
/// still accept the unknown field. The day one of them closes, that assertion
/// fails and it moves into the table above.
const OPEN_ON_WRITE: &[Probe] = &[
    Probe {
        resource: "guardrail_attachment",
        pointer: "",
        document: || json!({"guardrail_id": "gid", "scope_type": "env", "priority": 1}),
    },
    Probe {
        resource: "cache_policy",
        pointer: "",
        document: || json!({"name": "c"}),
    },
];

/// The field name every probe inserts. Long and unmistakable so a failure
/// message says which key the schema tripped over.
const PROBE: &str = "from_a_newer_control_plane";

/// Insert [`PROBE`] into the object `pointer` selects (the root for `""`).
fn probed(mut document: Value, pointer: &str) -> Value {
    let target = document
        .pointer_mut(pointer)
        .unwrap_or_else(|| panic!("pointer {pointer:?} resolves in the fixture"))
        .as_object_mut()
        .unwrap_or_else(|| panic!("pointer {pointer:?} selects an object"));
    target.insert(PROBE.to_string(), json!(1));
    document
}

#[test]
fn published_lenient_schemas_tolerate_what_the_strict_ones_reject() {
    let compile = |dir: &str, resource: &str| {
        jsonschema::validator_for(&published_schema(dir, resource))
            .unwrap_or_else(|e| panic!("schemas/{dir}/{resource}.schema.json compiles: {e}"))
    };

    for probe in UNKNOWN_FIELD_TOLERANCE {
        let resource = probe.resource;
        let pointer = probe.pointer;
        let base = (probe.document)();
        let with_unknown = probed(base.clone(), pointer);
        let strict = compile("resources", resource);
        let lenient = compile("resources-lenient", resource);

        if let Err(e) = strict.validate(&base) {
            panic!("{resource}: the fixture is not a valid document to begin with: {e}");
        }
        assert!(
            strict.validate(&with_unknown).is_err(),
            "{resource}: the strict file accepted `{PROBE}` at {pointer:?} — \
             that position is no longer closed on write"
        );
        if let Err(e) = lenient.validate(&with_unknown) {
            panic!(
                "{resource}: the lenient file rejected `{PROBE}` at {pointer:?}, so the \
                 loader would skip a row a newer control plane wrote: {e}"
            );
        }
    }

    for probe in OPEN_ON_WRITE {
        let resource = probe.resource;
        let with_unknown = probed((probe.document)(), probe.pointer);
        assert!(
            compile("resources", resource)
                .validate(&with_unknown)
                .is_ok(),
            "{resource} now closes unknown fields on write — move it into \
             UNKNOWN_FIELD_TOLERANCE with a pointer at the closed position"
        );
        assert!(compile("resources-lenient", resource)
            .validate(&with_unknown)
            .is_ok());
    }

    // Exhaustive over the resource list, so a new resource cannot be added
    // without deciding which of the two tables it belongs in.
    let mut covered: Vec<&str> = UNKNOWN_FIELD_TOLERANCE
        .iter()
        .chain(OPEN_ON_WRITE)
        .map(|p| p.resource)
        .collect();
    covered.sort_unstable();
    let mut all = RESOURCES.to_vec();
    all.sort_unstable();
    assert_eq!(covered, all);
}
