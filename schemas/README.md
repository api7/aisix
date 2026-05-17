# aisix canonical JSON Schemas

This directory holds canonical JSON Schema files for `aisix-core` resource
types. The files are **auto-generated** from the Rust type definitions in
`crates/aisix-core/src/models/` — do not edit them by hand.

## Layout

```text
schemas/
└── resources/
    ├── api_key.schema.json
    ├── cache_policy.schema.json
    ├── guardrail.schema.json
    ├── model.schema.json
    ├── observability_exporter.schema.json
    ├── provider_key.schema.json
    ├── rate_limit.schema.json
    ├── rate_limit_policy.schema.json
    └── routing.schema.json
```

Each file is a self-contained JSON Schema draft-07 document. Nested
types (e.g. `Adapter`, `RoutingTarget`, `TelemetryTags`) live in the
`definitions/` section of the parent resource — no cross-file `$ref` is
emitted.

File names use the snake_case singular form of the Rust type
(`api_key.schema.json`, `provider_key.schema.json`). The corresponding
etcd key prefix uses the plural `Resource::kind()` value
(`api_keys`, `provider_keys`); the two naming conventions are
deliberately distinct because the schema file is a per-type artifact
while the etcd prefix groups a collection of instances.

## Forward-compatibility

Three top-level resources intentionally **omit**
`additionalProperties: false`:

- `guardrail.schema.json` — the discriminated-union `kind` field uses
  serde's `flatten + tag` pattern, which is incompatible with a strict
  outer deny; strict typo-rejection happens earlier via
  `aisix-core::models::schema::validate_guardrail`.
- `cache_policy.schema.json` — cp-api may ship forward-compat fields
  ahead of a DP rollout, e.g. a new backend variant.
- `observability_exporter.schema.json` — same forward-compat reason as
  `cache_policy`.

Downstream consumers that default to strict validation should permit
unknown keys for these three resources; the other six are strict.

## Regenerating

After modifying any resource struct in `crates/aisix-core/src/models/`,
re-run:

```bash
cargo run -p aisix-core --bin dump-schema
```

CI runs the same command and fails the build if `schemas/` drifts from
the Rust types (drift-check workflow, separate PR).

## Downstream consumers

- `crates/aisix-admin/src/openapi.rs` — DP admin OpenAPI 3.1 document.
  Refactor target: replace inline schema objects with `$ref` into these
  files. (Follow-up PR.)
- `api7/AISIX-Cloud` cp-api — pulls these files (via submodule or
  pinned tag) for REST input validation against the same shape DP
  consumes from etcd.
- `api7/AISIX-Cloud` dashboard — renders forms straight from these
  schemas with [RJSF](https://github.com/rjsf-team/react-jsonschema-form)
  or equivalent, instead of hand-coded validators.

Refs api7/ai-gateway#304 item #1 (canonical JSON Schema as config
source of truth).
