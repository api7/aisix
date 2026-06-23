//! Characterization (golden-corpus) tests for the resource validators that
//! were migrated from hand-written `json!` schemas to struct-derived schemas
//! (single source of truth). Each corpus pins the exact accept/reject behavior
//! so the migration can prove it preserved the config contract (except the
//! documented intended changes — e.g. rate_limit `rps`/`rph` on api_key).
//!
//! One table per resource; the label is printed on failure so the offending
//! case is obvious. New resources append their own table as they migrate.

use aisix_core::models::schema::{
    validate_apikey, validate_cache_policy, validate_guardrail, validate_guardrail_attachment,
    validate_observability_exporter, validate_provider_key, validate_rate_limit_policy,
};
use serde_json::{json, Value};

/// Run a corpus of `(label, expect_accept, payload)` against `validate`.
#[track_caller]
fn check(
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
            ("missing allowed_models", false, json!({"key_hash": "h"})),
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
                "unknown window enum",
                false,
                json!({"name": "q", "scope": "team", "scope_ref": "t1", "window": "day", "max_requests": 1}),
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
    check(
        validate_provider_key,
        &[
            (
                "minimal",
                true,
                json!({"display_name": "openai-prod", "secret": "sk-x"}),
            ),
            (
                "with api_base + provider",
                true,
                json!({"display_name": "p", "secret": "sk-x", "api_base": "https://api.openai.com/v1", "provider": "deepseek"}),
            ),
            ("missing display_name", false, json!({"secret": "sk-x"})),
            ("missing secret", false, json!({"display_name": "x"})),
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
