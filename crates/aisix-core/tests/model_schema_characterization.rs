//! Characterization (golden-corpus) test for the `model` resource validator.
//!
//! This pins the EXACT accept/reject behavior of `validate_model` so that the
//! single-source-of-truth refactor (deriving the runtime validator from the
//! `Model` struct + schemars instead of a hand-written `json!` schema) can
//! prove it did not silently widen or narrow the config contract.
//!
//! Every case below is a constraint that the hand-written `model_schema()`
//! enforces today. After the refactor, this corpus must stay green — with one
//! deliberate, documented exception: the `rate_limit.rps` / `rate_limit.rph`
//! cases (see `INTENTIONALLY_FLIPPED` below), which the hand-written validator
//! wrongly rejects even though the `RateLimit` struct and the dispatch path
//! support them. The refactor flips those to ACCEPT and the assertions move
//! accordingly, surfacing the behavior change as a visible diff.

use aisix_core::models::schema::validate_model;
use serde_json::{json, Value};

/// Assert that the current `validate_model` ACCEPTS `value`.
#[track_caller]
fn accept(label: &str, value: Value) {
    if let Err(e) = validate_model(&value) {
        panic!("expected ACCEPT for `{label}`, got reject: {e}");
    }
}

/// Assert that the current `validate_model` REJECTS `value`.
#[track_caller]
fn reject(label: &str, value: Value) {
    if validate_model(&value).is_ok() {
        panic!("expected REJECT for `{label}`, but it was accepted");
    }
}

// ---------------------------------------------------------------------------
// ACCEPT — the three valid model shapes and their optional fields.
// ---------------------------------------------------------------------------

#[test]
fn accept_direct_minimal() {
    accept(
        "direct minimal",
        json!({
            "display_name": "m",
            "provider": "openai",
            "model_name": "gpt-4o",
            "provider_key_id": "pk-1"
        }),
    );
}

#[test]
fn accept_direct_full() {
    accept(
        "direct with every optional field",
        json!({
            "display_name": "m",
            "provider": "openai",
            "model_name": "gpt-4o",
            "provider_key_id": "pk-1",
            "timeout": 30000,
            "stream_timeout": 2500,
            "rate_limit": {"tpm": 1, "tpd": 1, "rpm": 1, "rpd": 1, "concurrency": 1},
            "allowed_cidrs": ["10.0.0.0/8"],
            "cost": {"input_per_1k": 0.0, "output_per_1k": 1.5},
            "background_model_check": {
                "enabled": true,
                "interval_seconds": 5,
                "timeout_seconds": 1,
                "prompt": "ok",
                "max_tokens": 1,
                "ignore_statuses": [408, 429],
                "stale_after_seconds": 1
            },
            "cooldown": {
                "enabled": true,
                "default_seconds": 0,
                "max_seconds": 1,
                "honor_retry_after": true,
                "trigger_statuses": [429, 503],
                "trigger_on_timeout": true,
                "trigger_on_transport": true
            }
        }),
    );
}

#[test]
fn accept_provider_with_dot() {
    // #417 regression guard: real models.dev ids like `wafer.ai` contain a dot.
    accept(
        "provider wafer.ai",
        json!({
            "display_name": "m",
            "provider": "wafer.ai",
            "model_name": "x",
            "provider_key_id": "pk-1"
        }),
    );
}

#[test]
fn accept_routing_minimal() {
    accept(
        "routing minimal",
        json!({
            "display_name": "r",
            "routing": {"targets": [{"model": "m"}]}
        }),
    );
}

#[test]
fn accept_routing_full() {
    accept(
        "routing with all knobs + shared optionals",
        json!({
            "display_name": "r",
            "routing": {
                "strategy": "weighted",
                "targets": [{"model": "a", "weight": 3}, {"model": "b", "weight": 1}],
                "retries": 2,
                "max_fallbacks": 1,
                "retry_on_429": true,
                "on_all_filtered": "original_order"
            },
            "timeout": 1000,
            "rate_limit": {"rpm": 10},
            "allowed_cidrs": ["10.0.0.0/8"],
            "cost": {"input_per_1k": 0.0, "output_per_1k": 0.0}
        }),
    );
}

#[test]
fn accept_ensemble_minimal() {
    accept(
        "ensemble minimal",
        json!({
            "display_name": "e",
            "ensemble": {"panel": [{"model": "m"}], "judge": {"model": "j"}}
        }),
    );
}

#[test]
fn accept_ensemble_full() {
    accept(
        "ensemble with panel + judge knobs",
        json!({
            "display_name": "e",
            "ensemble": {
                "panel": [{"model": "a", "temperature": 0.0, "seed": 0, "weight": 1}],
                "judge": {"model": "j", "synthesis_prompt": "synthesize"},
                "min_responses": 1,
                "timeout_ms": 0
            }
        }),
    );
}

#[test]
fn accept_direct_with_cooldown_and_background_check() {
    // The direct branch permits cooldown + background_model_check; routing and
    // ensemble do not. Locks the asymmetry.
    accept(
        "direct + cooldown + background_model_check",
        json!({
            "display_name": "m",
            "provider": "openai",
            "model_name": "gpt-4o",
            "provider_key_id": "pk-1",
            "cooldown": {"enabled": true},
            "background_model_check": {
                "enabled": true,
                "interval_seconds": 5,
                "timeout_seconds": 1,
                "prompt": "ok",
                "max_tokens": 1,
                "stale_after_seconds": 1
            }
        }),
    );
}

// ---------------------------------------------------------------------------
// REJECT — top-level shape.
// ---------------------------------------------------------------------------

#[test]
fn reject_missing_display_name() {
    reject(
        "missing display_name",
        json!({"provider": "openai", "model_name": "g", "provider_key_id": "pk-1"}),
    );
}

#[test]
fn reject_empty_display_name() {
    reject(
        "empty display_name",
        json!({"display_name": "", "provider": "openai", "model_name": "g", "provider_key_id": "pk-1"}),
    );
}

#[test]
fn reject_unknown_top_level_field() {
    reject(
        "unknown top-level field",
        json!({"display_name": "m", "provider": "openai", "model_name": "g", "provider_key_id": "pk-1", "foo": 1}),
    );
}

// ---------------------------------------------------------------------------
// REJECT — provider pattern / length (these constraints exist ONLY in the
// hand-written validator today; they must survive the refactor).
// ---------------------------------------------------------------------------

#[test]
fn reject_provider_uppercase() {
    reject(
        "provider uppercase",
        json!({"display_name": "m", "provider": "OpenAI", "model_name": "g", "provider_key_id": "pk-1"}),
    );
}

#[test]
fn reject_provider_leading_punctuation() {
    reject(
        "provider leading dot",
        json!({"display_name": "m", "provider": ".openai", "model_name": "g", "provider_key_id": "pk-1"}),
    );
    reject(
        "provider leading dash",
        json!({"display_name": "m", "provider": "-openai", "model_name": "g", "provider_key_id": "pk-1"}),
    );
}

#[test]
fn reject_provider_empty() {
    reject(
        "provider empty string",
        json!({"display_name": "m", "provider": "", "model_name": "g", "provider_key_id": "pk-1"}),
    );
}

#[test]
fn reject_provider_too_long() {
    reject(
        "provider > 64 chars",
        json!({"display_name": "m", "provider": "a".repeat(65), "model_name": "g", "provider_key_id": "pk-1"}),
    );
}

// ---------------------------------------------------------------------------
// REJECT — numeric / type bounds.
// ---------------------------------------------------------------------------

#[test]
fn reject_negative_timeout() {
    reject(
        "negative timeout",
        json!({"display_name": "m", "provider": "openai", "model_name": "g", "provider_key_id": "pk-1", "timeout": -1}),
    );
}

#[test]
fn reject_non_integer_timeout() {
    reject(
        "fractional timeout",
        json!({"display_name": "m", "provider": "openai", "model_name": "g", "provider_key_id": "pk-1", "timeout": 1.5}),
    );
}

#[test]
fn reject_empty_allowed_cidr_entry() {
    reject(
        "allowed_cidrs with empty entry",
        json!({"display_name": "m", "provider": "openai", "model_name": "g", "provider_key_id": "pk-1", "allowed_cidrs": [""]}),
    );
}

// ---------------------------------------------------------------------------
// REJECT — oneOf mutual exclusion (the constraint schemars cannot express from
// a flat struct; the refactor re-adds it via `#[schemars(extend(...))]`).
// ---------------------------------------------------------------------------

#[test]
fn reject_no_shape_at_all() {
    reject("display_name only — no direct/routing/ensemble", json!({"display_name": "m"}));
}

#[test]
fn reject_partial_direct() {
    reject(
        "provider without model_name/provider_key_id",
        json!({"display_name": "m", "provider": "openai"}),
    );
}

#[test]
fn reject_routing_plus_provider() {
    reject(
        "routing + provider",
        json!({"display_name": "m", "provider": "openai", "routing": {"targets": [{"model": "x"}]}}),
    );
}

#[test]
fn reject_routing_plus_ensemble() {
    reject(
        "routing + ensemble",
        json!({
            "display_name": "m",
            "routing": {"targets": [{"model": "x"}]},
            "ensemble": {"panel": [{"model": "a"}], "judge": {"model": "j"}}
        }),
    );
}

#[test]
fn reject_routing_plus_cooldown() {
    reject(
        "routing + cooldown (cooldown is direct-only)",
        json!({"display_name": "m", "routing": {"targets": [{"model": "x"}]}, "cooldown": {"enabled": true}}),
    );
}

#[test]
fn reject_routing_plus_background_check() {
    reject(
        "routing + background_model_check (direct-only)",
        json!({
            "display_name": "m",
            "routing": {"targets": [{"model": "x"}]},
            "background_model_check": {
                "enabled": true, "interval_seconds": 5, "timeout_seconds": 1,
                "prompt": "ok", "max_tokens": 1, "stale_after_seconds": 1
            }
        }),
    );
}

#[test]
fn reject_ensemble_plus_provider() {
    reject(
        "ensemble + provider",
        json!({
            "display_name": "m",
            "provider": "openai",
            "ensemble": {"panel": [{"model": "a"}], "judge": {"model": "j"}}
        }),
    );
}

// ---------------------------------------------------------------------------
// REJECT — nested object constraints.
// ---------------------------------------------------------------------------

#[test]
fn reject_routing_empty_targets() {
    reject(
        "routing targets empty",
        json!({"display_name": "r", "routing": {"targets": []}}),
    );
}

#[test]
fn reject_routing_target_missing_model() {
    reject(
        "routing target without model",
        json!({"display_name": "r", "routing": {"targets": [{"weight": 1}]}}),
    );
}

#[test]
fn reject_routing_unknown_field() {
    reject(
        "routing unknown field",
        json!({"display_name": "r", "routing": {"targets": [{"model": "x"}], "bogus": 1}}),
    );
}

#[test]
fn reject_routing_bad_strategy() {
    reject(
        "routing invalid strategy",
        json!({"display_name": "r", "routing": {"strategy": "random", "targets": [{"model": "x"}]}}),
    );
}

#[test]
fn reject_routing_bad_on_all_filtered() {
    reject(
        "routing invalid on_all_filtered",
        json!({"display_name": "r", "routing": {"targets": [{"model": "x"}], "on_all_filtered": "shrug"}}),
    );
}

#[test]
fn reject_ensemble_missing_judge() {
    reject(
        "ensemble without judge",
        json!({"display_name": "e", "ensemble": {"panel": [{"model": "a"}]}}),
    );
}

#[test]
fn reject_ensemble_empty_panel() {
    reject(
        "ensemble empty panel",
        json!({"display_name": "e", "ensemble": {"panel": [], "judge": {"model": "j"}}}),
    );
}

#[test]
fn reject_cost_missing_field() {
    reject(
        "cost missing output_per_1k",
        json!({"display_name": "m", "provider": "openai", "model_name": "g", "provider_key_id": "pk-1", "cost": {"input_per_1k": 1.0}}),
    );
}

#[test]
fn reject_cost_negative() {
    reject(
        "cost negative input_per_1k",
        json!({"display_name": "m", "provider": "openai", "model_name": "g", "provider_key_id": "pk-1", "cost": {"input_per_1k": -1.0, "output_per_1k": 0.0}}),
    );
}

#[test]
fn reject_background_check_missing_required() {
    reject(
        "background_model_check missing prompt",
        json!({
            "display_name": "m", "provider": "openai", "model_name": "g", "provider_key_id": "pk-1",
            "background_model_check": {"enabled": true, "interval_seconds": 5, "timeout_seconds": 1, "max_tokens": 1, "stale_after_seconds": 1}
        }),
    );
}

#[test]
fn reject_background_check_interval_too_small() {
    reject(
        "background_model_check interval_seconds < 5",
        json!({
            "display_name": "m", "provider": "openai", "model_name": "g", "provider_key_id": "pk-1",
            "background_model_check": {"enabled": true, "interval_seconds": 4, "timeout_seconds": 1, "prompt": "ok", "max_tokens": 1, "stale_after_seconds": 1}
        }),
    );
}

#[test]
fn reject_background_check_status_out_of_range() {
    reject(
        "background_model_check ignore_statuses below 100",
        json!({
            "display_name": "m", "provider": "openai", "model_name": "g", "provider_key_id": "pk-1",
            "background_model_check": {"enabled": true, "interval_seconds": 5, "timeout_seconds": 1, "prompt": "ok", "max_tokens": 1, "ignore_statuses": [99], "stale_after_seconds": 1}
        }),
    );
    reject(
        "background_model_check ignore_statuses above 599",
        json!({
            "display_name": "m", "provider": "openai", "model_name": "g", "provider_key_id": "pk-1",
            "background_model_check": {"enabled": true, "interval_seconds": 5, "timeout_seconds": 1, "prompt": "ok", "max_tokens": 1, "ignore_statuses": [600], "stale_after_seconds": 1}
        }),
    );
}

#[test]
fn reject_cooldown_unknown_field() {
    reject(
        "cooldown unknown field",
        json!({
            "display_name": "m", "provider": "openai", "model_name": "g", "provider_key_id": "pk-1",
            "cooldown": {"enabled": true, "bogus": 1}
        }),
    );
}

// ---------------------------------------------------------------------------
// INTENTIONALLY FLIPPED by the single-source refactor.
//
// The hand-written validator's `$defs/rate_limit` lists only 5 fields
// (tpm/tpd/rpm/rpd/concurrency) with `additionalProperties: false`, so it
// rejects `rps`/`rph` — even though the `RateLimit` struct declares them and
// the rate limiter honors per-second / per-hour windows (#426). These two
// assertions encode TODAY's (buggy) behavior; the refactor commit flips them
// to `accept(...)` and adds an E2E proving the limiter enforces them.
// ---------------------------------------------------------------------------

#[test]
fn reject_rate_limit_rps_today() {
    reject(
        "rate_limit.rps (flips to ACCEPT in the refactor)",
        json!({"display_name": "m", "provider": "openai", "model_name": "g", "provider_key_id": "pk-1", "rate_limit": {"rps": 10}}),
    );
}

#[test]
fn reject_rate_limit_rph_today() {
    reject(
        "rate_limit.rph (flips to ACCEPT in the refactor)",
        json!({"display_name": "m", "provider": "openai", "model_name": "g", "provider_key_id": "pk-1", "rate_limit": {"rph": 10}}),
    );
}
