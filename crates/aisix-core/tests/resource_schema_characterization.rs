//! Characterization (golden-corpus) tests for the resource validators that
//! were migrated from hand-written `json!` schemas to struct-derived schemas
//! (single source of truth). Each corpus pins the exact accept/reject behavior
//! so the migration can prove it preserved the config contract (except the
//! documented intended changes — e.g. rate_limit `rps`/`rph` on api_key).
//!
//! One table per resource; the label is printed on failure so the offending
//! case is obvious. New resources append their own table as they migrate.

use aisix_core::models::schema::{
    validate_apikey, validate_cache_policy, validate_rate_limit_policy,
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
