# aisix canonical JSON Schemas

This directory holds canonical JSON Schema files for `aisix-core` resource
types. The files are **auto-generated** from the Rust type definitions in
`crates/aisix-core/src/models/` — do not edit them by hand.

## Layout

```text
schemas/
├── resources/            # strict — the write contract
│   ├── api_key.schema.json
│   ├── cache_policy.schema.json
│   ├── embedding.schema.json
│   ├── guardrail.schema.json
│   ├── model.schema.json
│   ├── observability_exporter.schema.json
│   ├── provider_key.schema.json
│   ├── rate_limit.schema.json
│   ├── rate_limit_policy.schema.json
│   ├── routing.schema.json
│   └── semantic.schema.json
└── resources-lenient/    # lenient — the etcd read contract, same file names
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

## Strictness: these files describe the write contract

The published schemas carry the **write contract** for self-managed
declarative writes: `aisix validate --resources` checks a file offline,
and the `resources_file` source enforces the same shape at boot and
SIGHUP. A payload that fails them — including unknown fields, where a
resource closes them — is rejected on those paths. They are generated
from the same producers the in-repo strict validators compile, so the
published files and the gateway's own validators cannot drift.

Two write paths sit outside this enforcement: the AISIX Cloud control
plane validates requests against its own API schema before writing
etcd, and a **raw direct etcd put gets no synchronous validation** —
the document is only checked on read, by the lenient loader below.
Validate documents before putting them.

The gateway's **etcd read path is deliberately more lenient** (#871): a
stored document carrying fields outside these schemas still loads, with
the unknown fields ignored and reported as partially compatible on
`GET /status/config`, the heartbeat, and the
`aisix_config_partially_compatible_resources` metric. This keeps an
older gateway serving documents written by a newer control plane. Every
other constraint in these files — types, required fields, ranges, closed
enum value sets — applies on both paths.

That read contract is published too, as `resources-lenient/` — see below.

## `resources-lenient/`: what this build will LOAD

`resources-lenient/` carries the same resources under the same file
names, generated from the same producers with `strict: false` — the
exact schemas the etcd snapshot loader compiles into `LENIENT_SCHEMAS`
and validates every stored document against. A consumer that needs to
know what a given gateway release will accept from etcd reads these
files rather than deriving them from the strict ones.

The two sets differ in exactly one way: a lenient file carries no
`additionalProperties: false`, at **any** depth — not on the root, not on
a `definitions` entry, not on a `oneOf` branch, not on a nested property.
Field names, types, `required` lists, ranges, enum value sets, the
`$ref`/`definitions` layout and the `if`/`then`/`oneOf` structure are
identical. That single difference is the whole tolerance: an optional
field a newer control plane adds inside a nested config object is
ignored and reported, instead of taking the whole row down.

The five nested struct types (`ensemble`, `rate_limit`, `routing`,
`semantic`, `embedding`) have no standalone loader validator — they are
only ever validated as part of the resource that embeds them — so their
lenient files are the same struct-derived schema run through the same
opening pass.

Three top-level resources intentionally **omit**
`additionalProperties: false` even on the write contract:

- `guardrail.schema.json` — the discriminated-union `kind` field uses
  serde's `flatten + tag` pattern, which is incompatible with a strict
  outer deny; strict typo-rejection happens earlier via
  `aisix-core::models::schema::validate_guardrail`.
- `cache_policy.schema.json` — historically open on write as well.
- `observability_exporter.schema.json` — the top level is open, but the
  per-`kind` branches stay closed on both paths: an unknown field there
  could smuggle a plaintext credential past the `credential_ref`
  indirection, and serde cannot report ignored fields inside the
  tagged union, so an open branch would be a silent tolerance.

## Regenerating

After modifying any resource struct in `crates/aisix-core/src/models/`,
re-run:

```bash
cargo run -p aisix-core --bin dump-schema
```

After modifying Admin API routes, OpenAPI metadata, or the generated
resource schemas, verify that the Admin API OpenAPI generator still
emits a valid document:

```bash
cargo run -p aisix-admin --bin dump-openapi > /tmp/admin-api.openapi.json
```

CI runs the resource-schema drift check and the Admin API OpenAPI
generation check.

Release builds publish the Admin API OpenAPI document to
`/ai-gateway/openapi-<version>.json` and `/ai-gateway/openapi-latest.json`
on the configured `run.api7.ai` bucket. Main-branch builds publish
`/ai-gateway/openapi-dev.json` when the S3 and CloudFront secrets are
configured in the repository.

## Downstream consumers

- `crates/aisix-admin/src/openapi.rs` — DP admin OpenAPI 3.1 document.
  Refactor target: replace inline schema objects with `$ref` into these
  files. (Follow-up PR.)
- Documentation sites can consume the hosted Admin API OpenAPI document
  for the AISIX AI Gateway Admin API reference.
- Control-plane services can pin these files for REST input validation
  against the same shape the data plane consumes from etcd, and pin
  `resources-lenient/` to reason about what an already-deployed gateway
  release will still load.
- Dashboards can render forms from these schemas with
  [RJSF](https://github.com/rjsf-team/react-jsonschema-form) or
  equivalent, instead of hand-coded validators.

Refs api7/ai-gateway#304 item #1 (canonical JSON Schema as config
source of truth).
