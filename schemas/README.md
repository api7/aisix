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
│   ├── guardrail.schema.json
│   ├── model.schema.json
│   ├── provider_key.schema.json
│   └── …                 # one per resource, plus the nested struct types
└── resources-lenient/    # lenient — the etcd read contract, same file names
```

Both directories hold the same file names. The listing above is a sample;
the set is whatever `dump-schema` emits, which is every entry of
`schema::RESOURCES` plus `ensemble`, `rate_limit`, `routing`, `semantic`
and `embedding`.

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

Four top-level resources intentionally **omit**
`additionalProperties: false` even on the write contract (the list is
`closes_on_write` in `crates/aisix-core/src/models/schema.rs`):

- `guardrail.schema.json` — the discriminated-union `kind` field uses
  serde's `flatten + tag` pattern, which is incompatible with a strict
  outer deny; strict typo-rejection happens earlier via
  `aisix-core::models::schema::validate_guardrail`.
- `cache_policy.schema.json` — historically open on write as well.
- `guardrail_attachment.schema.json` — likewise.
- `observability_exporter.schema.json` — the top level is open, but the
  per-`kind` branches stay closed on the write path, so a misspelled
  field is rejected rather than dropped. On the read path they open like
  every other closure (see `resources-lenient/` below); the tolerance is
  not silent there because serde cannot report ignored fields inside a
  tagged union, so the loader takes this resource's unknown-field report
  from the strict schema instead (`unknown_field_paths`).

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

**Do not validate writes against these files.** They are deliberately
open, and for four resources they relax more than that (below), so a
consumer that swaps `resources/` for `resources-lenient/` in a
vendoring script silently turns its input validation into an
accept-almost-anything gate. `resources/` stays the schema for anything
a user submits; `resources-lenient/` answers only "will this build load
this stored document".

### How the two sets differ

For **every** resource, a lenient file carries no
`additionalProperties: false`, at **any** depth — not on the root, not on
a `definitions` entry, not on a `oneOf` branch, not on a nested property.
That is the tolerance the split exists for: an optional field a newer
control plane adds inside a nested config object is ignored and reported,
instead of taking the whole row down.

For **four** resources the read contract relaxes a requirement as well,
so a consumer that models the lenient set as "the strict set with
`additionalProperties` stripped" is wrong about them:

| resource | additionally relaxed on read |
| --- | --- |
| `api_key` | `McpAccess.allow` is not required |
| `guardrail` | the `semantic` kind requires neither `embedding_model` nor a threshold beside each example list |
| `mcp_policy` | `allow` is not required |
| `model` | the per-kind `not`/`anyOf` lists that forbid a knob a kind never resolves are shorter — a stored row keeps loading and `Model::strip_kind_inapplicable` drops the dead knob |

Note what is NOT in that table: the `custom` guardrail's `script` is
required on **both** sets. A scriptless `custom` row screens nothing
either way, so rejecting it is what makes it visible in
`GET /status/config`'s `rejected[]`.

These come from the four producers that take a `strict` flag in
`crates/aisix-core/src/models/schema.rs` and are deliberate.

Separately, the lenient files keep three `default` annotations the
strict producer strips on purpose — `default: 0.75` on the `semantic`
guardrail's `allow_threshold`/`deny_threshold`, and `default: ""` on the
`custom` kind's `script`, which sits beside `minLength: 1`. They change
nothing about what validates, but a form generator that honours them
pre-fills a threshold the operator was deliberately asked to choose, or
a script value the same branch refuses. Generate forms from
`resources/`.

The exact paths at which the two sets diverge are pinned by
`published_sets_differ_only_where_registered` in
`crates/aisix-core/tests/resource_schema_characterization.rs`, so a new
divergence — or a change to one of these — has to be registered before
the suite goes green. Everything else is identical: field names, types,
ranges, enum value sets, the `$ref`/`definitions` layout, and the
`if`/`then`/`oneOf` structure.

### Two gates sit behind the lenient schema

Passing a lenient file is necessary, not sufficient. After the schema
gate the loader still deserialises the document into the Rust type, and
a value the schema does not constrain (an integer past `u64`, say) fails
there and takes the row; and `rate_limit_policy` runs a semantic pass
(`validate_semantics`) for cross-field rules JSON Schema cannot express,
which also rejects a row whole. Treat these files as the necessary
condition for a row to load, not the complete one.

### The five nested struct types are documentation, not a contract

`ensemble`, `rate_limit`, `routing`, `semantic` and `embedding` have no
standalone validator on either path — they are only ever validated as
part of the resource that embeds them — so their standalone files, in
**both** sets, document the struct's shape rather than anything that is
enforced. They are also generated with schemars' default `Option<T>`
rendering, which the embedding resources do not all use: `rate_limit`
standalone renders `rpm` as `["integer", "null"]`, and
`model.schema.json#/definitions/RateLimit` renders it as `"integer"`, so
a `model` document writing an explicit `null` there is accepted by the
standalone file and skipped by the loader. The authoritative copy of a
nested type is always
`<parent>.schema.json#/definitions/<Type>` — read it there. (Their key
order also differs between the two sets, since only the lenient side
round-trips through a sorted JSON map.)

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
- Control-plane services can pin `resources/` for REST input validation
  against the same shape the data plane consumes from etcd, and pin
  `resources-lenient/` to reason about what an already-deployed gateway
  release will still load — never the other way round.
- Dashboards can render forms from `resources/` with
  [RJSF](https://github.com/rjsf-team/react-jsonschema-form) or
  equivalent, instead of hand-coded validators — from `resources/` and
  not its lenient twin, which keeps `default` annotations the write
  contract deliberately drops (above).

Refs api7/ai-gateway#304 item #1 (canonical JSON Schema as config
source of truth).
