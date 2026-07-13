//! Pipeline tests for the resources-file source. Everything goes
//! through [`load_from_str`] with a closed env map — no process-global
//! environment mutation, no filesystem.

use super::*;
use std::collections::HashMap;

fn env_of(pairs: &[(&str, &str)]) -> HashMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

fn load(contents: &str, env: &HashMap<String, String>) -> Result<AisixSnapshot, FileSourceErrors> {
    load_from_str(contents, "resources.yaml", 1, &|name| {
        env.get(name).cloned()
    })
}

fn errors_of(result: Result<AisixSnapshot, FileSourceErrors>) -> Vec<String> {
    result
        .expect_err("expected load errors")
        .errors
        .iter()
        .map(ToString::to_string)
        .collect()
}

const FULL_VALID_FILE: &str = r#"
_format_version: "1"

provider_keys:
  - display_name: openai-prod
    provider: openai
    api_key: ${OPENAI_API_KEY}
    api_base: https://${UPSTREAM_HOST}/v1

models:
  - display_name: gpt-4o
    provider: openai
    model_name: gpt-4o-2024-11-20
    provider_key: openai-prod
  - display_name: gpt-4o-mini
    provider: openai
    model_name: gpt-4o-mini
    provider_key: openai-prod
  - display_name: balanced
    routing:
      strategy: round_robin
      targets:
        - model: gpt-4o
        - model: gpt-4o-mini

api_keys:
  - display_name: ci-bot
    key_env: E2E_CI_BOT_KEY
    allowed_models: ["gpt-4o", "balanced"]
  - display_name: ops
    key_hash: 91ed2dbc407561556f3e7be98ba0bd2a57986d6a868c482d867d19c6d40d201c
    allowed_models: ["*"]

guardrails:
  - name: no-secrets
    kind: keyword
    patterns:
      - kind: literal
        value: topsecret

mcp_servers:
  - name: github
    url: https://mcp.example.com/mcp

a2a_agents:
  - name: helper
    url: https://a2a.example.com/agent

cache_policies:
  - name: default-cache
    enabled: true
    ttl_seconds: 600

observability_exporters:
  - name: otel
    kind: otlp_http
    endpoint: https://otel.example.com/v1/traces

rate_limit_policies:
  - name: cap-gpt4o
    scope: model
    scope_ref: gpt-4o
    window: minute
    max_requests: 300
  - name: cap-ci-bot
    scope: api_key
    scope_ref: ci-bot
    window: minute
    max_requests: 60
  - name: cap-team
    scope: team
    scope_ref: team-uuid-1
    window: hour
    max_requests: 1000
"#;

fn full_env() -> HashMap<String, String> {
    env_of(&[
        ("OPENAI_API_KEY", "sk-upstream"),
        ("UPSTREAM_HOST", "api.openai.com"),
        ("E2E_CI_BOT_KEY", "sk-ci-plaintext"),
    ])
}

#[test]
fn full_valid_file_loads_every_kind() {
    let snap = load(FULL_VALID_FILE, &full_env()).expect("full file must load");
    assert_eq!(snap.provider_keys.len(), 1);
    assert_eq!(snap.models.len(), 3);
    assert_eq!(snap.apikeys.len(), 2);
    assert_eq!(snap.guardrails.len(), 1);
    assert_eq!(snap.mcp_servers.len(), 1);
    assert_eq!(snap.a2a_agents.len(), 1);
    assert_eq!(snap.cache_policies.len(), 1);
    assert_eq!(snap.observability_exporters.len(), 1);
    assert_eq!(snap.rate_limit_policies.len(), 3);

    // Interpolation landed in the provider key (full + partial).
    let pk = snap.provider_keys.get_by_name("openai-prod").unwrap();
    assert_eq!(pk.value.api_key, "sk-upstream");
    assert_eq!(
        pk.value.api_base.as_deref(),
        Some("https://api.openai.com/v1")
    );

    // The model's provider_key name sugar resolved to the derived id.
    let model = snap.models.get_by_name("gpt-4o").unwrap();
    assert_eq!(
        model.value.provider_key_id.as_deref(),
        Some(derive_id("provider_keys", "openai-prod").as_str()),
    );
    assert_eq!(model.id, derive_id("models", "gpt-4o"));

    // key_env became the SHA-256 of the plaintext; the api_keys name
    // index is keyed by key_hash (matching the etcd path).
    let expected_hash = crate::models::ApiKey::hash_bearer("sk-ci-plaintext");
    let key = snap
        .apikeys
        .get_by_name(&expected_hash)
        .expect("hashed key");
    assert_eq!(key.id, derive_id("api_keys", "ci-bot"));

    // scope_ref resolution per scope: model / api_key → derived ids,
    // team → verbatim.
    let by_policy_name = |n: &str| {
        snap.rate_limit_policies
            .entries()
            .into_iter()
            .find(|e| e.value.name == n)
            .unwrap()
    };
    assert_eq!(
        by_policy_name("cap-gpt4o").value.scope_ref,
        derive_id("models", "gpt-4o"),
    );
    assert_eq!(
        by_policy_name("cap-ci-bot").value.scope_ref,
        derive_id("api_keys", "ci-bot"),
    );
    assert_eq!(by_policy_name("cap-team").value.scope_ref, "team-uuid-1");
}

#[test]
fn ids_are_deterministic_across_two_loads() {
    let env = full_env();
    let a = load(FULL_VALID_FILE, &env).unwrap();
    let b = load(FULL_VALID_FILE, &env).unwrap();
    for name in ["gpt-4o", "gpt-4o-mini", "balanced"] {
        assert_eq!(
            a.models.get_by_name(name).unwrap().id,
            b.models.get_by_name(name).unwrap().id,
            "model {name} id must be stable across reloads",
        );
    }
    assert_eq!(
        a.provider_keys.get_by_name("openai-prod").unwrap().id,
        b.provider_keys.get_by_name("openai-prod").unwrap().id,
    );
}

#[test]
fn missing_format_version_is_a_load_error() {
    let errs = errors_of(load("models: []\n", &env_of(&[])));
    assert_eq!(errs.len(), 1, "{errs:?}");
    assert!(
        errs[0].contains("missing mandatory _format_version"),
        "{errs:?}"
    );
}

#[test]
fn unrecognized_format_version_is_a_load_error() {
    let errs = errors_of(load("_format_version: \"2\"\n", &env_of(&[])));
    assert!(errs[0].contains("unrecognized _format_version"), "{errs:?}");
    // An unquoted `1` parses as a YAML integer — the error nudges toward
    // quoting instead of claiming the version is missing.
    let errs = errors_of(load("_format_version: 1\n", &env_of(&[])));
    assert!(errs[0].contains("quote it"), "{errs:?}");
}

#[test]
fn unknown_top_level_key_is_a_load_error() {
    let errs = errors_of(load("_format_version: \"1\"\nmodles: []\n", &env_of(&[])));
    assert_eq!(errs.len(), 1, "{errs:?}");
    assert!(
        errs[0].contains("unknown top-level key `modles`"),
        "{errs:?}"
    );
    assert!(
        errs[0].contains("provider_keys"),
        "should list the collections: {errs:?}"
    );
}

#[test]
fn empty_and_multi_document_files_are_load_errors() {
    let errs = errors_of(load("", &env_of(&[])));
    assert!(errs[0].contains("file is empty"), "{errs:?}");

    let errs = errors_of(load(
        "_format_version: \"1\"\n---\n_format_version: \"1\"\n",
        &env_of(&[]),
    ));
    assert!(errs[0].contains("single YAML document"), "{errs:?}");
}

#[test]
fn interpolation_error_names_kind_entry_and_field_path() {
    let contents = r#"
_format_version: "1"
provider_keys:
  - display_name: pk
    api_key: ${MISSING_KEY}
"#;
    let errs = errors_of(load(contents, &env_of(&[])));
    assert_eq!(errs.len(), 1, "{errs:?}");
    assert!(errs[0].starts_with("provider_keys[0]"), "{errs:?}");
    assert!(errs[0].contains("field `api_key`"), "{errs:?}");
    assert!(errs[0].contains("`MISSING_KEY`"), "{errs:?}");
}

#[test]
fn duplicate_identity_within_a_kind_is_a_load_error() {
    let contents = r#"
_format_version: "1"
provider_keys:
  - display_name: dup
    api_key: sk-1
  - display_name: dup
    api_key: sk-2
"#;
    let errs = errors_of(load(contents, &env_of(&[])));
    assert_eq!(errs.len(), 1, "{errs:?}");
    assert!(
        errs[0].contains("duplicate provider_keys entry"),
        "{errs:?}"
    );
    assert!(
        errs[0].contains("provider_keys[0]"),
        "should name the first definition: {errs:?}"
    );
}

#[test]
fn id_field_is_rejected_on_strict_and_open_kinds() {
    // `guardrails` is one of the schema-open kinds — without the
    // explicit sugar-layer check an `id` would be silently carried.
    let contents = r#"
_format_version: "1"
guardrails:
  - name: g
    id: 11111111-1111-1111-1111-111111111111
    kind: keyword
    patterns:
      - kind: literal
        value: x
models:
  - display_name: m
    id: 22222222-2222-2222-2222-222222222222
    provider: openai
    model_name: x
    provider_key_id: pk-1
"#;
    let errs = errors_of(load(contents, &env_of(&[])));
    assert_eq!(errs.len(), 2, "{errs:?}");
    for e in &errs {
        assert!(e.contains("does not accept `id`"), "{errs:?}");
    }
}

#[test]
fn canonical_validation_failures_carry_entry_scope() {
    // Empty display_name violates the model schema (minLength 1)…
    // after passing identity extraction? No — identity extraction
    // requires a non-empty string, so this surfaces as the identity
    // error. Use a schema-level failure instead: a direct model missing
    // its provider triple.
    let contents = r#"
_format_version: "1"
models:
  - display_name: incomplete
    provider: openai
"#;
    let errs = errors_of(load(contents, &env_of(&[])));
    assert_eq!(errs.len(), 1, "{errs:?}");
    assert!(
        errs[0].starts_with("models[0] (\"incomplete\")"),
        "{errs:?}"
    );
    assert!(errs[0].contains("schema validation failed"), "{errs:?}");
}

#[test]
fn all_errors_are_collected_not_first_only() {
    let contents = r#"
_format_version: "1"
provider_keys:
  - display_name: pk
    api_key: ${MISSING_A}
models:
  - display_name: m1
    provider: openai
    model_name: x
    provider_key: no-such-pk
api_keys:
  - display_name: k1
    key_env: MISSING_B
    allowed_models: ["ghost-model"]
"#;
    let errs = errors_of(load(contents, &env_of(&[])));
    // 1 interpolation error + 1 unknown provider_key name + 1 missing
    // key_env variable. (`ghost-model` is unreachable for k1 because its
    // entry already failed desugaring — cross-ref only runs on decoded
    // entries; the load still fails with everything found.)
    assert_eq!(errs.len(), 3, "{errs:?}");
    assert!(errs.iter().any(|e| e.contains("MISSING_A")), "{errs:?}");
    assert!(errs.iter().any(|e| e.contains("no-such-pk")), "{errs:?}");
    assert!(errs.iter().any(|e| e.contains("MISSING_B")), "{errs:?}");
}

#[test]
fn cross_ref_unknown_allowed_model_is_an_error_globs_exempt() {
    let contents = r#"
_format_version: "1"
provider_keys:
  - display_name: pk
    api_key: sk-1
models:
  - display_name: real-model
    provider: openai
    model_name: x
    provider_key: pk
api_keys:
  - display_name: globby
    key_hash: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    allowed_models: ["*", "openai/*", "real-model"]
  - display_name: typo
    key_hash: bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
    allowed_models: ["real-modle"]
"#;
    let errs = errors_of(load(contents, &env_of(&[])));
    assert_eq!(errs.len(), 1, "globs and exact matches must pass: {errs:?}");
    assert!(errs[0].contains("api_keys[1]"), "{errs:?}");
    assert!(errs[0].contains("\"real-modle\""), "{errs:?}");
    assert!(
        errs[0].contains("real-model"),
        "should list defined models: {errs:?}"
    );
}

#[test]
fn cross_ref_covers_routing_ensemble_and_semantic_references() {
    let contents = r#"
_format_version: "1"
provider_keys:
  - display_name: pk
    api_key: sk-1
models:
  - display_name: real
    provider: openai
    model_name: x
    provider_key: pk
  - display_name: router
    routing:
      strategy: round_robin
      targets:
        - model: real
        - model: ghost-target
  - display_name: council
    ensemble:
      panel:
        - model: real
        - model: ghost-panel
      judge:
        model: ghost-judge
  - display_name: sem
    semantic:
      embedding_model: ghost-embed
      default: ghost-default
      routes:
        - name: r1
          target: ghost-route
          examples: ["hi"]
      match:
        threshold: 0.7
"#;
    let errs = errors_of(load(contents, &env_of(&[])));
    let all = errs.join("\n");
    for ghost in [
        "ghost-target",
        "ghost-panel",
        "ghost-judge",
        "ghost-embed",
        "ghost-default",
        "ghost-route",
    ] {
        assert!(
            all.contains(ghost),
            "missing cross-ref error for {ghost}:\n{all}"
        );
    }
    // `real` referenced from routing/ensemble passed the check.
    assert_eq!(errs.len(), 6, "{errs:?}");
}

#[test]
fn duplicate_key_hash_across_api_keys_is_a_load_error_without_hash_leak() {
    // The runtime credential index is keyed by key_hash — a duplicate
    // plaintext would silently last-wins at auth time, so it must fail
    // the load like the duplicate-identity rule does. One entry uses
    // key_env, the other key_hash, resolving to the same credential.
    let plain = "sk-shared-plaintext";
    let hash = crate::models::ApiKey::hash_bearer(plain);
    let contents = format!(
        r#"
_format_version: "1"
api_keys:
  - display_name: first
    key_env: SHARED_KEY
    allowed_models: ["*"]
  - display_name: second
    key_hash: {hash}
    allowed_models: ["*"]
"#
    );
    let env = env_of(&[("SHARED_KEY", plain)]);
    let errs = errors_of(load(&contents, &env));
    assert_eq!(errs.len(), 1, "{errs:?}");
    assert!(errs[0].contains("duplicate api key credential"), "{errs:?}");
    // The error names both entries…
    assert!(errs[0].contains("api_keys[0]"), "{errs:?}");
    assert!(errs[0].contains("api_keys[1]"), "{errs:?}");
    // …and never echoes the credential in either form.
    assert!(!errs[0].contains(plain), "plaintext leaked: {errs:?}");
    assert!(!errs[0].contains(&hash), "hash leaked: {errs:?}");
}

#[test]
fn explicit_provider_key_id_must_match_a_file_defined_key() {
    // In file mode every provider-key id is derived from its name, so a
    // pasted foreign UUID is guaranteed dangling — reject it at load.
    let contents = r#"
_format_version: "1"
provider_keys:
  - display_name: pk
    api_key: sk-1
models:
  - display_name: m1
    provider: openai
    model_name: x
    provider_key_id: 11111111-1111-1111-1111-111111111111
"#;
    let errs = errors_of(load(contents, &env_of(&[])));
    assert_eq!(errs.len(), 1, "{errs:?}");
    assert!(errs[0].contains("provider_key_id"), "{errs:?}");
    assert!(
        errs[0].contains("`provider_key`"),
        "should point at the name sugar: {errs:?}"
    );

    // The correctly-derived id (what the sugar would produce) passes —
    // determinism means a generated file can carry explicit ids.
    let derived = derive_id("provider_keys", "pk");
    let contents = format!(
        r#"
_format_version: "1"
provider_keys:
  - display_name: pk
    api_key: sk-1
models:
  - display_name: m1
    provider: openai
    model_name: x
    provider_key_id: {derived}
"#
    );
    let snap = load(&contents, &env_of(&[])).expect("derived id must be accepted");
    assert_eq!(
        snap.models
            .get_by_name("m1")
            .unwrap()
            .value
            .provider_key_id
            .as_deref(),
        Some(derived.as_str()),
    );
}

#[test]
fn absent_collections_load_as_empty_and_null_sections_are_tolerated() {
    let contents = "_format_version: \"1\"\nmodels:\n";
    let snap = load(contents, &env_of(&[])).unwrap();
    assert_eq!(snap.total_entries(), 0);
}

#[test]
fn collection_must_be_a_sequence() {
    let errs = errors_of(load(
        "_format_version: \"1\"\nmodels: {display_name: x}\n",
        &env_of(&[]),
    ));
    assert!(errs[0].contains("`models` must be a sequence"), "{errs:?}");
}

#[test]
fn entry_must_be_a_mapping() {
    let errs = errors_of(load(
        "_format_version: \"1\"\nmodels:\n  - just-a-string\n",
        &env_of(&[]),
    ));
    assert!(errs[0].contains("models[0]"), "{errs:?}");
    assert!(errs[0].contains("must be a mapping"), "{errs:?}");
}

#[test]
fn mcp_servers_accept_display_name_as_alternative_identity() {
    let contents = r#"
_format_version: "1"
mcp_servers:
  - display_name: gh-former
    url: https://x.example/mcp
"#;
    let snap = load(contents, &env_of(&[])).unwrap();
    // The alias lands on `name` through the same serde path as etcd.
    assert!(snap.mcp_servers.get_by_name("gh-former").is_some());
    assert_eq!(
        snap.mcp_servers.get_by_name("gh-former").unwrap().id,
        derive_id("mcp_servers", "gh-former"),
    );
}

#[test]
fn revision_is_stamped_on_every_entry() {
    let env = full_env();
    let snap = load_from_str(FULL_VALID_FILE, "resources.yaml", 7, &|n| {
        env.get(n).cloned()
    })
    .unwrap();
    assert_eq!(snap.models.get_by_name("gpt-4o").unwrap().revision, 7);
    assert_eq!(
        snap.provider_keys
            .get_by_name("openai-prod")
            .unwrap()
            .revision,
        7
    );
}

#[test]
fn report_formats_file_and_all_errors() {
    let err = load("models: []\n", &env_of(&[])).unwrap_err();
    let text = err.to_string();
    assert!(text.contains("resources file resources.yaml"), "{text}");
    assert!(text.contains("1 error(s)"), "{text}");
    assert!(
        text.contains("  - (file): missing mandatory _format_version"),
        "{text}"
    );
}
