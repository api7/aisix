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
//! Two validator sets exist since issue #871 (strict write / lenient read):
//!
//! - **Strict** ([`SCHEMAS`], the plain `validate_*` functions): unknown
//!   fields are rejected (where a resource closes them). Used by the
//!   in-repo declarative writers — `aisix validate` and the file
//!   source — so typos keep failing loud, and published as the
//!   resource schema files in `schemas/resources/` — the write
//!   contract. (The control plane validates against its own API
//!   schema; raw direct etcd puts are only checked on read.)
//! - **Lenient** ([`LENIENT_SCHEMAS`], the `validate_*_lenient` functions):
//!   unknown fields pass; every other constraint (types, required, ranges,
//!   closed enums) still applies. Used only by the etcd snapshot loader so a
//!   document written by a newer control plane loads with its extra fields
//!   ignored — and reported — instead of whole-row rejected.
//!
//! Both sets build from the same per-resource producers; strictness is a
//! mechanical pair of passes over the produced value — [`close_unknown_fields`]
//! for the write set, [`open_unknown_fields`] for the read set — so the two
//! can never drift field-wise. Every closure, whether the write pass placed it
//! or a producer injected it by hand (the `observability_exporter` and
//! guardrail kind branches, the guardrail tagged sub-enums, the untagged
//! `ConditionNode`/`OnEmbeddingFailure` variants), belongs to the write set
//! alone: the read pass strips `additionalProperties: false` at EVERY depth,
//! so an optional field a newer control plane added inside a nested config
//! object is ignored instead of taking the whole row down with it.
//!
//! Tolerated is not the same as silent. Unknown fields inside serde-buffered
//! content — everything under the guardrail/exporter `#[serde(flatten)]`ed
//! tagged config, or inside an untagged variant — never reach
//! `serde_ignored`, so the loader takes their paths from
//! [`unknown_field_paths`], which reads them off the strict schema.
//!
//! The watch path reuses step 2 on incoming events — malformed payloads are
//! skipped with a warning and do not take down the gateway.

use crate::models::model::Model;
use jsonschema::Validator;
use once_cell::sync::Lazy;
use serde_json::{json, Value};
use std::sync::Arc;
use thiserror::Error;

/// Cached compiled schemas. Compiling on every write would be wasteful; the
/// schemas are static, so we build them once. This is the **strict** set:
/// unknown fields fail validation wherever the resource model closes them.
pub struct Schemas {
    pub model: Validator,
    pub apikey: Validator,
    pub provider_key: Validator,
    pub guardrail: Validator,
    pub guardrail_attachment: Validator,
    pub cache_policy: Validator,
    pub observability_exporter: Validator,
    pub rate_limit_policy: Validator,
    pub mcp_server: Validator,
    pub mcp_policy: Validator,
    pub a2a_agent: Validator,
    pub oidc_provider: Validator,
    pub claim_mapping: Validator,
    pub passthrough_route: Validator,
    pub mcp_auth_settings: Validator,
}

pub static SCHEMAS: Lazy<Arc<Schemas>> = Lazy::new(|| Arc::new(Schemas::compile(true)));

/// The **lenient** twin of [`SCHEMAS`]: same producers, without the
/// [`close_unknown_fields`] pass. Only the etcd snapshot loader validates
/// against this set (issue #871); every write path stays on [`SCHEMAS`].
pub static LENIENT_SCHEMAS: Lazy<Arc<Schemas>> = Lazy::new(|| Arc::new(Schemas::compile(false)));

/// Every resource with a runtime validator, by the name
/// [`resource_root_schema`] takes. The published schema files and the
/// validator sets are built from this list, so a new resource cannot reach
/// one without reaching the other.
pub const RESOURCES: [&str; 15] = [
    "model",
    "api_key",
    "provider_key",
    "guardrail",
    "guardrail_attachment",
    "cache_policy",
    "observability_exporter",
    "rate_limit_policy",
    "mcp_server",
    "mcp_policy",
    "a2a_agent",
    "oidc_provider",
    "claim_mapping",
    "passthrough_route",
    "mcp_auth_settings",
];

/// Whether a resource's write contract closes unknown top-level fields.
/// `cache_policy`, `guardrail`, `guardrail_attachment` and
/// `observability_exporter` historically ship open root schemas (documented
/// on their producers), so the strict closing pass skips them. Shared by
/// [`Schemas::compile`] and the resource schema published by `dump-schema`,
/// so the enforced write contract and the published one cannot drift.
fn closes_on_write(resource: &str) -> bool {
    !matches!(
        resource,
        "cache_policy" | "guardrail" | "guardrail_attachment" | "observability_exporter"
    )
}

/// The canonical schema of one resource, as enforced on the given path.
/// `strict` selects the write contract (unknown fields rejected wherever the
/// resource closes them); `!strict` the etcd read contract (unknown fields
/// tolerated). This is the single producer both validator sets and the
/// `dump-schema` binary build from.
pub fn resource_root_schema(resource: &str, strict: bool) -> Value {
    let mut schema = match resource {
        "model" => model_root_schema(strict),
        "api_key" => apikey_root_schema(strict),
        "provider_key" => provider_key_root_schema(),
        "guardrail" => guardrail_root_schema(),
        "guardrail_attachment" => guardrail_attachment_root_schema(),
        "cache_policy" => cache_policy_root_schema(),
        "observability_exporter" => observability_exporter_root_schema(),
        "rate_limit_policy" => rate_limit_policy_root_schema(),
        "mcp_server" => mcp_server_root_schema(),
        "mcp_policy" => mcp_policy_root_schema(strict),
        "a2a_agent" => a2a_agent_root_schema(),
        "oidc_provider" => oidc_provider_root_schema(),
        "claim_mapping" => claim_mapping_root_schema(),
        "passthrough_route" => passthrough_route_root_schema(),
        "mcp_auth_settings" => mcp_auth_settings_root_schema(),
        other => panic!("unknown resource {other:?}"),
    };
    if strict {
        if closes_on_write(resource) {
            close_unknown_fields(&mut schema);
        }
    } else {
        open_unknown_fields(&mut schema);
    }
    schema
}

impl Schemas {
    fn compile(strict: bool) -> Self {
        let build = |resource: &str| {
            jsonschema::options()
                .build(&resource_root_schema(resource, strict))
                .unwrap_or_else(|e| panic!("{resource} schema is well-formed: {e}"))
        };
        Self {
            model: build("model"),
            apikey: build("api_key"),
            provider_key: build("provider_key"),
            guardrail: build("guardrail"),
            guardrail_attachment: build("guardrail_attachment"),
            cache_policy: build("cache_policy"),
            observability_exporter: build("observability_exporter"),
            rate_limit_policy: build("rate_limit_policy"),
            mcp_server: build("mcp_server"),
            mcp_policy: build("mcp_policy"),
            a2a_agent: build("a2a_agent"),
            oidc_provider: build("oidc_provider"),
            claim_mapping: build("claim_mapping"),
            passthrough_route: build("passthrough_route"),
            mcp_auth_settings: build("mcp_auth_settings"),
        }
    }
}

/// Close a produced resource schema against unknown fields: insert
/// `additionalProperties: false` on the root object and on every
/// `definitions` entry that is a plain object schema (has `properties`).
///
/// This reproduces exactly what `#[serde(deny_unknown_fields)]` made
/// `schemars` emit before issue #871 moved strictness out of the structs:
///
/// - conditional/overlay subschemas (`oneOf`/`anyOf`/`allOf`/`if` branches)
///   are never touched — closing an `if`/`then` overlay would reject every
///   field the overlay does not list;
/// - an existing `additionalProperties` value is preserved, whether the
///   deliberate `false` on hand-closed branches or the value schema of a
///   map-typed field;
/// - enum-shaped definitions (no `properties`) are skipped.
pub fn close_unknown_fields(schema: &mut Value) {
    close_object(schema);
    close_definitions(schema);
}

/// Insert `additionalProperties: false` on one schema node, if it is a plain
/// object schema that has not already stated its own answer.
fn close_object(node: &mut Value) {
    let Some(obj) = node.as_object_mut() else {
        return;
    };
    if obj.contains_key("properties") && !obj.contains_key("additionalProperties") {
        obj.insert("additionalProperties".to_string(), json!(false));
    }
}

/// Close every struct-shaped `definitions` entry, leaving the root alone.
/// Used on its own by the producers of resources whose ROOT stays open on the
/// write path but whose nested structs do not (`guardrail`).
fn close_definitions(schema: &mut Value) {
    if let Some(Value::Object(defs)) = schema.get_mut("definitions") {
        for def in defs.values_mut() {
            close_object(def);
        }
    }
}

/// Open a produced resource schema against unknown fields at EVERY depth:
/// drop `additionalProperties: false` wherever it sits — the root, a
/// `definitions` entry, a `oneOf`/`anyOf` branch, a nested property.
///
/// Not a mirror of [`close_unknown_fields`], deliberately. Closing is a
/// shallow, hand-placed decision (closing an `if`/`then` overlay would reject
/// every field the overlay does not list), so it touches only the root and
/// the definitions. Opening has to reach everywhere a closure can be written
/// — including the ones `schemars` renders from a nested
/// `#[serde(deny_unknown_fields)]` struct and the ones a producer injects by
/// hand — because a closure the read path leaves standing is a whole stored
/// row lost the first time a newer control plane adds a field under it.
///
/// The read set is therefore free of `additionalProperties: false` by
/// construction, which is what makes "an additive optional field is ignored
/// by older data planes" true at every nesting depth instead of only at the
/// document root.
pub fn open_unknown_fields(schema: &mut Value) {
    match schema {
        Value::Object(obj) => {
            if obj.get("additionalProperties") == Some(&Value::Bool(false)) {
                obj.remove("additionalProperties");
            }
            for child in obj.values_mut() {
                open_unknown_fields(child);
            }
        }
        Value::Array(items) => {
            for item in items {
                open_unknown_fields(item);
            }
        }
        _ => {}
    }
}

/// The resources whose unknown-field report cannot come from
/// `serde_ignored`, mapped to their strict schema.
///
/// `serde_ignored` reports a field only when the ignored-field callback
/// fires, and it never fires inside serde-buffered content: a
/// `#[serde(flatten)]`ed internally-tagged enum (the entire `guardrail` and
/// `observability_exporter` config subtree) and untagged variants
/// (`ConditionNode`, `OnEmbeddingFailure`) are deserialised from a buffered
/// `Content`, out of the wrapper's sight. Those are exactly the places the
/// read schema now leaves open, so without a second source the tolerance
/// would be silent.
///
/// A resource listed here must not carry a schema-hidden but
/// serde-consumed field — a `#[schemars(skip)]` tombstone like
/// `McpAccess::legacy_mode`. This walk reads known field names off the
/// schema, so a field the schema does not mention would be reported as
/// unknown on every row that carries it.
static REPORT_SCHEMAS: Lazy<Vec<(&'static str, Value)>> = Lazy::new(|| {
    [
        "guardrail",
        "observability_exporter",
        "model",
        "rate_limit_policy",
    ]
    .into_iter()
    .map(|resource| (resource, resource_root_schema(resource, true)))
    .collect()
});

/// Paths of the fields `value` carries that this build's write contract for
/// `resource` does not know, at every depth — the loader's unknown-field
/// report for the resources `serde_ignored` cannot see into
/// ([`REPORT_SCHEMAS`]). Returns an empty vector for every other resource.
///
/// `jsonschema` cannot stand in for this: a `oneOf` failure collapses to one
/// root-level "not valid under any of the schemas" error that names no field,
/// and every guardrail and exporter document is a `oneOf`.
///
/// Conservative by construction. A key is reported only when NO branch
/// applicable at that position declares it, so cross-kind leakage (a
/// `datadog` exporter carrying an `otlp_http` field) does not show up here —
/// that is a malformed document, not a field from a newer build, and the
/// write path rejects it. Map-valued fields (`headers`,
/// `severity_threshold_by_category`) declare a value schema rather than
/// properties, so their keys are never unknown.
pub fn unknown_field_paths(resource: &str, value: &Value) -> Vec<String> {
    let Some((_, schema)) = REPORT_SCHEMAS.iter().find(|(name, _)| *name == resource) else {
        return Vec::new();
    };
    let empty = serde_json::Map::new();
    let definitions = schema
        .get("definitions")
        .and_then(Value::as_object)
        .unwrap_or(&empty);
    let mut out = Vec::new();
    collect_unknown_fields(&[schema], definitions, value, "", &mut out);
    out
}

/// Flatten the schema nodes that apply to one value position: the node
/// itself, whatever its `$ref` names, and every `allOf`/`oneOf`/`anyOf`
/// member. Branch selection is deliberately skipped — the union of what all
/// branches declare is what keeps [`unknown_field_paths`] conservative.
fn expand_applicable<'a>(
    nodes: &[&'a Value],
    definitions: &'a serde_json::Map<String, Value>,
) -> Vec<&'a Value> {
    let mut pending: Vec<&Value> = nodes.to_vec();
    let mut applicable = Vec::new();
    while let Some(node) = pending.pop() {
        let Some(obj) = node.as_object() else {
            continue;
        };
        if let Some(name) = obj
            .get("$ref")
            .and_then(Value::as_str)
            .and_then(|r| r.strip_prefix("#/definitions/"))
        {
            if let Some(target) = definitions.get(name) {
                pending.push(target);
            }
        }
        for combinator in ["allOf", "oneOf", "anyOf"] {
            if let Some(Value::Array(members)) = obj.get(combinator) {
                pending.extend(members.iter());
            }
        }
        if obj.contains_key("properties")
            || obj.contains_key("additionalProperties")
            || obj.contains_key("items")
        {
            applicable.push(node);
        }
    }
    applicable
}

fn collect_unknown_fields(
    nodes: &[&Value],
    definitions: &serde_json::Map<String, Value>,
    value: &Value,
    path: &str,
    out: &mut Vec<String>,
) {
    let applicable = expand_applicable(nodes, definitions);
    match value {
        Value::Object(fields) => {
            // Nothing here describes an object shape (a `true` schema, an
            // enum) — no field name can be called unknown against it.
            if !applicable.iter().any(|n| n.get("properties").is_some()) {
                return;
            }
            for (key, child) in fields {
                let mut child_nodes: Vec<&Value> = Vec::new();
                for node in &applicable {
                    if let Some(declared) = node.get("properties").and_then(|p| p.get(key)) {
                        child_nodes.push(declared);
                    } else if let Some(values) = node
                        .get("additionalProperties")
                        .filter(|v| v.is_object() || **v == Value::Bool(true))
                    {
                        child_nodes.push(values);
                    }
                }
                let child_path = if path.is_empty() {
                    key.clone()
                } else {
                    format!("{path}.{key}")
                };
                if child_nodes.is_empty() {
                    out.push(child_path);
                } else {
                    collect_unknown_fields(&child_nodes, definitions, child, &child_path, out);
                }
            }
        }
        Value::Array(items) => {
            let item_nodes: Vec<&Value> = applicable
                .iter()
                .filter_map(|node| node.get("items"))
                .collect();
            if item_nodes.is_empty() {
                return;
            }
            for (index, item) in items.iter().enumerate() {
                collect_unknown_fields(
                    &item_nodes,
                    definitions,
                    item,
                    &format!("{path}.{index}"),
                    out,
                );
            }
        }
        _ => {}
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
            // Mask instance values in the message. Validation errors flow
            // into logs, the rejection buffer surfaced upstream, and admin
            // 400 bodies — and resource documents carry credentials. The
            // renamed-field `anyOf` (see `accept_renamed_field`) sits at
            // the document root, so its unmasked message would echo the
            // whole stored document, credentials included.
            message: err.masked().to_string(),
        });
    }
    Ok(())
}

/// Strict model validation, with the per-kind dead-knob case named.
///
/// The model schema is a five-branch `oneOf` (one per kind), so when a
/// document carries a knob its kind never resolves, EVERY branch fails
/// and the first error `jsonschema` reports is the root-level "not valid
/// under any of the schemas" — true, but it does not say which field is
/// at fault. That is the one failure mode the strict path exists to
/// produce (`model_one_of_strict`), so it is worth naming.
///
/// The field list comes from [`Model::strip_kind_inapplicable`], the same
/// function the lenient loader uses to strip and report these knobs, so
/// the two paths cannot disagree about which knob is dead on which kind.
///
/// Best-effort by construction: a document that fails for any other
/// reason — an unknown field, a wrong type, a missing requirement — may
/// not deserialise at all, and keeps the generic message. Only the field
/// NAMES are added, never instance values, so this respects the masking
/// contract in [`validate`].
///
/// A dead knob is only reported when it is the WHOLE story: the document
/// is re-validated with exactly those fields removed, and the message is
/// replaced only if it then passes. A document that carries a dead knob
/// AND an independent violation keeps the original error, which points
/// at the other problem and is the more useful of the two — replacing it
/// would leave `path` and `message` describing different fields.
pub fn validate_model(value: &Value) -> Result<(), SchemaError> {
    let err = match validate(&SCHEMAS.model, value) {
        Ok(()) => return Ok(()),
        Err(err) => err,
    };
    let Ok(mut model) = serde_json::from_value::<Model>(value.clone()) else {
        return Err(err);
    };
    let dead = model.strip_kind_inapplicable();
    if dead.is_empty() {
        return Err(err);
    }
    // Probe the ORIGINAL document minus the dead fields rather than
    // re-serialising `model`: a serde round-trip drops unknown fields
    // and materialises defaults, either of which could make the probe
    // pass while the real document still fails. Every dead knob is a
    // top-level field.
    let mut probe = value.clone();
    match probe.as_object_mut() {
        Some(obj) => {
            for field in &dead {
                obj.remove(*field);
            }
        }
        None => return Err(err),
    }
    if validate(&SCHEMAS.model, &probe).is_err() {
        return Err(err);
    }
    // strip_kind_inapplicable only reports on these three kinds.
    let kind = if model.is_routing() {
        "model group"
    } else if model.is_ensemble() {
        "ensemble"
    } else {
        "semantic router"
    };
    Err(SchemaError {
        path: err.path,
        message: format!(
            "{} not accepted on a {kind}",
            dead.iter()
                .map(|f| format!("`{f}`"))
                .collect::<Vec<_>>()
                .join(", "),
        ),
    })
}

pub fn validate_apikey(value: &Value) -> Result<(), SchemaError> {
    validate(&SCHEMAS.apikey, value)
}

pub fn validate_provider_key(value: &Value) -> Result<(), SchemaError> {
    validate(&SCHEMAS.provider_key, value)
}

pub fn validate_guardrail(value: &Value) -> Result<(), SchemaError> {
    match validate(&SCHEMAS.guardrail, value) {
        Ok(()) => Ok(()),
        Err(err) => Err(name_unknown_fields(
            "guardrail",
            &SCHEMAS.guardrail,
            value,
            err,
        )),
    }
}

pub fn validate_cache_policy(value: &Value) -> Result<(), SchemaError> {
    validate(&SCHEMAS.cache_policy, value)
}

pub fn validate_observability_exporter(value: &Value) -> Result<(), SchemaError> {
    match validate(&SCHEMAS.observability_exporter, value) {
        Ok(()) => Ok(()),
        Err(err) => Err(name_unknown_fields(
            "observability_exporter",
            &SCHEMAS.observability_exporter,
            value,
            err,
        )),
    }
}

/// Replace a `oneOf` rejection with the unknown field names behind it.
///
/// `jsonschema` collapses every branch failure of a `oneOf` into one
/// root-level "not valid under any of the schemas" — true, but naming no
/// field, and a guardrail or exporter document IS a `oneOf`, so its most
/// ordinary write-path error (a typo) arrives unreadable. The strict schema
/// knows every field name, which is what [`unknown_field_paths`] reads.
///
/// Same discipline as [`validate_model`]: the names replace the message only
/// when they are the WHOLE story. The document is re-validated with exactly
/// those fields removed, and one that also violates something else keeps the
/// original error, which points at the other problem. Only field NAMES are
/// added, never instance values, so the masking contract in [`validate`]
/// holds.
fn name_unknown_fields(
    resource: &str,
    validator: &Validator,
    value: &Value,
    err: SchemaError,
) -> SchemaError {
    let unknown = unknown_field_paths(resource, value);
    if unknown.is_empty() {
        return err;
    }
    let mut probe = value.clone();
    for path in &unknown {
        remove_path(&mut probe, path);
    }
    if validate(validator, &probe).is_err() {
        return err;
    }
    SchemaError {
        path: err.path,
        message: format!(
            "unknown field(s): {}",
            unknown
                .iter()
                .map(|path| format!("`{path}`"))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

/// Remove one dotted path produced by [`unknown_field_paths`] from a
/// document; a numeric segment indexes into an array.
fn remove_path(value: &mut Value, path: &str) {
    let mut segments = path.split('.').peekable();
    let mut node = value;
    while let Some(segment) = segments.next() {
        if segments.peek().is_none() {
            if let Some(fields) = node.as_object_mut() {
                fields.remove(segment);
            }
            return;
        }
        node = match (node, segment.parse::<usize>()) {
            (Value::Array(items), Ok(index)) => match items.get_mut(index) {
                Some(item) => item,
                None => return,
            },
            (Value::Object(fields), _) => match fields.get_mut(segment) {
                Some(field) => field,
                None => return,
            },
            _ => return,
        };
    }
}

pub fn validate_rate_limit_policy(value: &Value) -> Result<(), SchemaError> {
    validate(&SCHEMAS.rate_limit_policy, value)
}

pub fn validate_guardrail_attachment(value: &Value) -> Result<(), SchemaError> {
    validate(&SCHEMAS.guardrail_attachment, value)
}

pub fn validate_mcp_server(value: &Value) -> Result<(), SchemaError> {
    validate(&SCHEMAS.mcp_server, value)
}

pub fn validate_a2a_agent(value: &Value) -> Result<(), SchemaError> {
    validate(&SCHEMAS.a2a_agent, value)
}

pub fn validate_mcp_policy(value: &Value) -> Result<(), SchemaError> {
    validate(&SCHEMAS.mcp_policy, value)
}

pub fn validate_oidc_provider(value: &Value) -> Result<(), SchemaError> {
    validate(&SCHEMAS.oidc_provider, value)
}

pub fn validate_claim_mapping(value: &Value) -> Result<(), SchemaError> {
    validate(&SCHEMAS.claim_mapping, value)
}

pub fn validate_passthrough_route(value: &Value) -> Result<(), SchemaError> {
    validate(&SCHEMAS.passthrough_route, value)
}

// ---- lenient variants (etcd snapshot loader only, issue #871) ----
//
// Unknown fields pass; every other constraint still applies. The loader
// pairs these with `serde_ignored` so tolerated fields are collected and
// reported as partially compatible rather than silently dropped.

pub fn validate_model_lenient(value: &Value) -> Result<(), SchemaError> {
    validate(&LENIENT_SCHEMAS.model, value)
}

pub fn validate_apikey_lenient(value: &Value) -> Result<(), SchemaError> {
    validate(&LENIENT_SCHEMAS.apikey, value)
}

pub fn validate_provider_key_lenient(value: &Value) -> Result<(), SchemaError> {
    validate(&LENIENT_SCHEMAS.provider_key, value)
}

pub fn validate_guardrail_lenient(value: &Value) -> Result<(), SchemaError> {
    validate(&LENIENT_SCHEMAS.guardrail, value)
}

pub fn validate_cache_policy_lenient(value: &Value) -> Result<(), SchemaError> {
    validate(&LENIENT_SCHEMAS.cache_policy, value)
}

pub fn validate_observability_exporter_lenient(value: &Value) -> Result<(), SchemaError> {
    validate(&LENIENT_SCHEMAS.observability_exporter, value)
}

pub fn validate_rate_limit_policy_lenient(value: &Value) -> Result<(), SchemaError> {
    validate(&LENIENT_SCHEMAS.rate_limit_policy, value)
}

pub fn validate_guardrail_attachment_lenient(value: &Value) -> Result<(), SchemaError> {
    validate(&LENIENT_SCHEMAS.guardrail_attachment, value)
}

pub fn validate_mcp_server_lenient(value: &Value) -> Result<(), SchemaError> {
    validate(&LENIENT_SCHEMAS.mcp_server, value)
}

pub fn validate_a2a_agent_lenient(value: &Value) -> Result<(), SchemaError> {
    validate(&LENIENT_SCHEMAS.a2a_agent, value)
}

pub fn validate_mcp_policy_lenient(value: &Value) -> Result<(), SchemaError> {
    validate(&LENIENT_SCHEMAS.mcp_policy, value)
}

pub fn validate_oidc_provider_lenient(value: &Value) -> Result<(), SchemaError> {
    validate(&LENIENT_SCHEMAS.oidc_provider, value)
}

pub fn validate_claim_mapping_lenient(value: &Value) -> Result<(), SchemaError> {
    validate(&LENIENT_SCHEMAS.claim_mapping, value)
}

pub fn validate_passthrough_route_lenient(value: &Value) -> Result<(), SchemaError> {
    validate(&LENIENT_SCHEMAS.passthrough_route, value)
}

pub fn validate_mcp_auth_settings(value: &Value) -> Result<(), SchemaError> {
    validate(&SCHEMAS.mcp_auth_settings, value)
}

pub fn validate_mcp_auth_settings_lenient(value: &Value) -> Result<(), SchemaError> {
    validate(&LENIENT_SCHEMAS.mcp_auth_settings, value)
}

/// Build a resource's canonical JSON Schema from its struct via `schemars`,
/// the single source of field shapes and per-field constraints.
///
/// `nullable_options` controls schemars' `Option<T>` representation: `false`
/// keeps optional fields plain-but-absent (`type: string`), matching the wire
/// shape of resources that never receive an explicit `null` (cp-api omits
/// unset fields); `true` keeps the default nullable form (`type: [string,
/// null]`) for resources whose schema deliberately accepts `null` (e.g.
/// ApiKey `team_id`/`user_id`).
///
/// Both the runtime validators in [`Schemas::compile`] and the `dump-schema`
/// binary that emits `schemas/resources/*.json` build from these producers, so
/// the published schema and the enforced schema are the same object by
/// construction — no hand-maintained second copy to drift.
fn struct_root_schema<T: schemars::JsonSchema>(nullable_options: bool) -> Value {
    use schemars::gen::{SchemaGenerator, SchemaSettings};

    let settings = SchemaSettings::draft07().with(|s| {
        s.option_add_null_type = nullable_options;
    });
    let root = SchemaGenerator::new(settings).into_root_schema_for::<T>();
    serde_json::to_value(root).expect("resource schema serializes to JSON")
}

/// Canonical JSON Schema for the `model` resource: the [`Model`] struct plus
/// the one cross-field invariant `schemars` cannot express
/// ([`super::model::model_one_of`] — the direct/routing/ensemble XOR).
/// `strict` picks the write-path variant that additionally forbids the
/// per-kind dead knobs ([`super::model::model_one_of_strict`]); the
/// lenient read path keeps the base XOR so stored rows load (and strip)
/// rather than drop.
///
/// [`Model`]: crate::models::Model
pub fn model_root_schema(strict: bool) -> Value {
    let mut schema = struct_root_schema::<crate::models::Model>(false);
    let one_of = if strict {
        super::model::model_one_of_strict()
    } else {
        super::model::model_one_of()
    };
    schema
        .as_object_mut()
        .expect("model root schema is a JSON object")
        .insert("oneOf".to_string(), one_of);
    // `OnEmbeddingFailure` is `#[serde(untagged)]` with an object variant
    // (`{ "target": … }`): serde buffers untagged content and silently
    // swallows unknown fields inside it, invisible to the write path's
    // serde step and to `serde_ignored` alike. The schema closure is
    // therefore the write path's only guard — and, once the read pass
    // opens it, what `unknown_field_paths` reads to keep the loader's
    // tolerance from being silent.
    if let Some(any_of) = schema
        .get_mut("definitions")
        .and_then(|d| d.get_mut("OnEmbeddingFailure"))
        .and_then(|b| b.get_mut("anyOf"))
        .and_then(Value::as_array_mut)
    {
        for branch in any_of.iter_mut() {
            if branch.get("type").and_then(Value::as_str) == Some("object") {
                if let Some(obj) = branch.as_object_mut() {
                    obj.insert("additionalProperties".to_string(), json!(false));
                }
            }
        }
    }
    schema
}

/// Canonical JSON Schema for the `api_key` resource, derived from the
/// [`ApiKey`](crate::models::ApiKey) struct. Uses the default nullable
/// `Option` representation so `team_id`/`user_id` keep accepting an explicit
/// `null` (cp-api sends `null` to clear team/owner), matching the resource's
/// wire contract.
pub fn apikey_root_schema(strict: bool) -> Value {
    let mut schema = struct_root_schema::<crate::models::ApiKey>(true);
    if strict {
        require_property(
            schema
                .pointer_mut("/definitions/McpAccess")
                .expect("api_key schema defines McpAccess"),
            "allow",
        );
    }
    schema
}

/// Add `name` to a schema object's `required` list, creating the list
/// when absent. Used for fields the WRITE path must see spelled out
/// while the runtime loader defaults them — a stale row has to keep
/// loading, but a new one must not acquire its meaning by omission.
fn require_property(schema: &mut Value, name: &str) {
    let obj = schema
        .as_object_mut()
        .expect("schema fragment is a JSON object");
    match obj.get_mut("required").and_then(Value::as_array_mut) {
        Some(list) => {
            if !list.iter().any(|v| v.as_str() == Some(name)) {
                list.push(json!(name));
            }
        }
        None => {
            obj.insert("required".to_string(), json!([name]));
        }
    }
}

/// Canonical JSON Schema for the `provider_key` resource, derived from the
/// [`ProviderKey`](crate::models::ProviderKey) struct. Uses the nullable
/// `Option` representation (`true`): `TelemetryTags` carries fields cp-api
/// sends as explicit `null` (`branded_provider`/`pk_label`/`byo_label`), and
/// keeping all optionals nullable matches the resource's wire contract.
/// The credential is accepted under both its canonical name `api_key` and
/// its former name `secret` (see [`accept_renamed_field`]).
pub fn provider_key_root_schema() -> Value {
    let mut schema = struct_root_schema::<crate::models::ProviderKey>(true);
    accept_renamed_field(
        &mut schema,
        "api_key",
        "secret",
        "Accepted as an alternative spelling of `api_key`. \
         Provide the credential under exactly one of the two names.",
    );
    schema
}

/// Mirror a struct field's `#[serde(alias = "…")]` in the generated schema.
///
/// `schemars` does not emit serde aliases, so a naively generated schema
/// would list only the canonical name and — with `additionalProperties:
/// false` — reject every stored document that still uses the former one at
/// the snapshot loader's schema gate. This transform makes the generated
/// schema accept both spellings:
///
/// - the former name is declared as a property with the same shape as the
///   canonical one (so `minLength` and type constraints keep applying);
/// - the canonical name is removed from `required` and requiredness becomes
///   a top-level `anyOf` of the two single-field `required` forms — at
///   least one spelling must be present;
/// - `additionalProperties: false` stays intact, so unknown fields are
///   still rejected.
///
/// A document carrying **both** spellings passes this schema and is then
/// rejected by serde's duplicate-field check at deserialize (both names map
/// to the same field), so the ambiguity never loads.
fn accept_renamed_field(schema: &mut Value, canonical: &str, former: &str, note: &str) {
    let obj = schema
        .as_object_mut()
        .expect("resource root schema is a JSON object");
    assert!(
        !obj.contains_key("anyOf"),
        "top-level anyOf already in use; compose the rename acceptance with allOf instead"
    );

    let properties = obj
        .get_mut("properties")
        .and_then(Value::as_object_mut)
        .expect("resource root schema has properties");
    let mut former_schema = properties
        .get(canonical)
        .unwrap_or_else(|| panic!("schema property `{canonical}` exists"))
        .clone();
    if let Some(former_obj) = former_schema.as_object_mut() {
        former_obj.insert("description".to_string(), Value::String(note.to_string()));
    }
    properties.insert(former.to_string(), former_schema);

    if let Some(Value::Array(required)) = obj.get_mut("required") {
        required.retain(|v| v.as_str() != Some(canonical));
    }
    // The branch titles label the two spellings in rendered references
    // (reference UIs use `title` for `anyOf` tab labels).
    obj.insert(
        "anyOf".to_string(),
        json!([
            { "title": canonical, "required": [canonical] },
            { "title": former, "required": [former] },
        ]),
    );
}

/// Canonical JSON Schema for the `mcp_server` resource, derived from the
/// [`McpServer`](crate::models::McpServer) struct. Uses the nullable `Option`
/// representation (`true`) so the optional fields (`secret`, `client_id`,
/// `token_url`, `scopes`, `timeout_ms`) accept an explicit `null` as well as
/// being absent, matching the resource's wire contract. The `transport` /
/// `auth_type` closed sets come from the
/// [`McpTransport`](crate::models::McpTransport) /
/// [`McpAuthType`](crate::models::McpAuthType) enums. The per-`auth_type`
/// credential coupling, and the openapi-only `spec`/`api_key_header` fields, are
/// injected here as an `allOf` of `if`/`then` subschemas (see
/// [`super::mcp_server::mcp_server_credential_coupling`]) so every configuration
/// path enforces them. The label
/// is accepted under both its canonical name `name` and its former name
/// `display_name` (see [`accept_renamed_field`]).
pub fn mcp_server_root_schema() -> Value {
    let mut schema = struct_root_schema::<crate::models::McpServer>(true);
    schema
        .as_object_mut()
        .expect("mcp server root schema is a JSON object")
        .insert(
            "allOf".to_string(),
            super::mcp_server::mcp_server_credential_coupling(),
        );
    accept_renamed_field(
        &mut schema,
        "name",
        "display_name",
        "Accepted as an alternative spelling of `name`. \
         Provide the label under exactly one of the two names.",
    );
    if let Some(Value::Object(defs)) = schema.get_mut("definitions") {
        title_single_value_enum_variants(
            defs,
            "McpAuthType",
            &[
                ("none", "No authentication"),
                ("bearer", "Bearer token"),
                ("api_key", "API key"),
                ("oauth2", "OAuth 2.0 client credentials"),
            ],
        );
        title_single_value_enum_variants(
            defs,
            "McpTransport",
            &[("streamable_http", "Streamable HTTP")],
        );
        title_single_value_enum_variants(
            defs,
            "McpServerType",
            &[
                ("mcp", "Upstream MCP server"),
                ("openapi", "REST API described by an OpenAPI document"),
            ],
        );
    }
    schema
}

/// Canonical JSON Schema for the `a2a_agent` resource, derived from the
/// [`A2aAgent`](crate::models::A2aAgent) struct. The label is accepted under
/// both its canonical name `name` and its former name `display_name` (see
/// [`accept_renamed_field`]).
pub fn a2a_agent_root_schema() -> Value {
    let mut schema = struct_root_schema::<crate::models::A2aAgent>(true);
    schema
        .as_object_mut()
        .expect("a2a agent root schema is a JSON object")
        .insert(
            "allOf".to_string(),
            super::a2a_agent::a2a_agent_credential_coupling(),
        );
    accept_renamed_field(
        &mut schema,
        "name",
        "display_name",
        "Accepted as an alternative spelling of `name`. \
         Provide the label under exactly one of the two names.",
    );
    if let Some(Value::Object(defs)) = schema.get_mut("definitions") {
        title_single_value_enum_variants(
            defs,
            "A2aAuthType",
            &[
                ("none", "No authentication"),
                ("bearer", "Bearer token"),
                ("api_key", "API key"),
            ],
        );
        title_single_value_enum_variants(
            defs,
            "A2aProtocolVersion",
            &[("1.0", "A2A 1.0"), ("0.3", "A2A 0.3")],
        );
    }
    schema
}

fn title_single_value_enum_variants(
    defs: &mut serde_json::Map<String, Value>,
    schema_name: &str,
    titles: &[(&str, &str)],
) {
    let Some(Value::Array(branches)) = defs.get_mut(schema_name).and_then(|d| d.get_mut("oneOf"))
    else {
        return;
    };
    for branch in branches.iter_mut() {
        let Some(branch) = branch.as_object_mut() else {
            continue;
        };
        let Some(value) = branch
            .get("enum")
            .and_then(|v| v.as_array())
            .and_then(|values| values.first())
            .and_then(|v| v.as_str())
        else {
            continue;
        };
        if let Some((_, title)) = titles.iter().find(|(expected, _)| *expected == value) {
            branch
                .entry("title".to_string())
                .or_insert_with(|| Value::String((*title).to_string()));
        }
    }
}

/// Canonical JSON Schema for the `oidc_provider` resource, derived from the
/// [`OidcProvider`](crate::models::OidcProvider) struct. Uses the
/// plain-but-absent `Option` representation (`false`): the control plane
/// omits unset fields (`jwks_uri`, `bound_claims`) rather than sending an
/// explicit `null`.
pub fn oidc_provider_root_schema() -> Value {
    let mut schema = struct_root_schema::<crate::models::OidcProvider>(false);
    // schemars does not propagate the `#[schemars(length(min = 1))]` on
    // the `BoundClaimExpect::Any(Vec<String>)` variant into the untagged
    // enum's array branch, so the generated schema would accept an empty
    // `bound_claims` value list. Re-assert `minItems: 1` to match the
    // model's non-empty contract.
    if let Some(any_of) = schema
        .get_mut("definitions")
        .and_then(|d| d.get_mut("BoundClaimExpect"))
        .and_then(|b| b.get_mut("anyOf"))
        .and_then(Value::as_array_mut)
    {
        for branch in any_of.iter_mut() {
            if branch.get("type").and_then(Value::as_str) == Some("array") {
                if let Some(obj) = branch.as_object_mut() {
                    obj.insert("minItems".to_string(), json!(1));
                }
            }
        }
    }
    schema
}

/// Canonical JSON Schema for the `claim_mapping` resource, derived from
/// the [`ClaimMapping`](crate::models::ClaimMapping) struct. Uses the
/// plain-but-absent `Option` representation (`false`): the resource has
/// no nullable fields, only defaults omitted when unset.
pub fn claim_mapping_root_schema() -> Value {
    let mut schema = struct_root_schema::<crate::models::ClaimMapping>(false);
    // `priority` has a stable runtime default of 0, but schemars drops
    // the `default` keyword for fields whose default value is skipped
    // on serialization (`skip_serializing_if = "is_zero"`). Re-assert
    // it so API consumers can discover the behavior from the contract,
    // matching the `enabled: true` default the derive does emit.
    if let Some(priority) = schema
        .get_mut("properties")
        .and_then(|p| p.get_mut("priority"))
        .and_then(Value::as_object_mut)
    {
        priority.insert("default".to_string(), json!(0));
    }
    schema
}

/// Canonical JSON Schema for the `passthrough_route` resource, derived from
/// the [`PassthroughRoute`](crate::models::PassthroughRoute) struct. Uses the
/// nullable `Option` representation (`true`) so unset optional fields accept
/// an explicit `null` as well as being absent. The `auth_mode` /
/// `credential_mode` closed sets come from their enums; every
/// cross-field invariant (match dimensions, target shape, per-mode required
/// companions) is injected as an `allOf` (see
/// [`super::passthrough_route::passthrough_route_coupling`]) so the strict
/// write path and the lenient etcd read path enforce the same coupling. The
/// label is accepted under both `name` and its `display_name` alias.
pub fn passthrough_route_root_schema() -> Value {
    let mut schema = struct_root_schema::<crate::models::PassthroughRoute>(true);
    schema
        .as_object_mut()
        .expect("passthrough route root schema is a JSON object")
        .insert(
            "allOf".to_string(),
            super::passthrough_route::passthrough_route_coupling(),
        );
    accept_renamed_field(
        &mut schema,
        "name",
        "display_name",
        "Accepted as an alternative spelling of `name`. \
         Provide the label under exactly one of the two names.",
    );
    if let Some(Value::Object(defs)) = schema.get_mut("definitions") {
        title_single_value_enum_variants(
            defs,
            "PassthroughAuthMode",
            &[
                ("gateway_key", "Standard gateway credential"),
                ("header_key", "Gateway credential in a dedicated header"),
                ("anonymous", "Anonymous (bound principal)"),
            ],
        );
        title_single_value_enum_variants(
            defs,
            "PassthroughCredentialMode",
            &[
                ("inject", "Inject the ProviderKey secret"),
                ("forward_client", "Forward the caller's own credential"),
            ],
        );
    }
    schema
}

/// Canonical JSON Schema for the `mcp_auth_settings` resource, derived
/// from the [`McpAuthSettings`](crate::models::McpAuthSettings) struct.
/// Both settings it carries (`resource_url` for OAuth discovery,
/// `anonymous` for credential-less access) are optional, and each is
/// self-contained — no cross-field coupling to inject.
/// The singleton-per-environment invariant is enforced by the writers
/// (cp-api keys the row by the environment id; the resources file
/// rejects duplicates at load) and, at read time, by the resolvers
/// failing closed — not by the document schema.
pub fn mcp_auth_settings_root_schema() -> Value {
    struct_root_schema::<crate::models::McpAuthSettings>(false)
}

/// Canonical JSON Schema for the `mcp_policy` resource, derived from the
/// [`McpPolicy`](crate::models::McpPolicy) struct. Uses the nullable `Option`
/// representation (`true`) so `scope_ref` accepts an explicit `null` as
/// well as being absent. The closed `scope` set comes from the
/// [`McpPolicyScope`](crate::models::McpPolicyScope) enum, plus the one
/// cross-field invariant `schemars` cannot express: a `team`-scoped policy
/// must name its team in `scope_ref` (otherwise the row could shadow the
/// environment layer).
pub fn mcp_policy_root_schema(strict: bool) -> Value {
    let mut schema = struct_root_schema::<crate::models::McpPolicy>(true);
    if strict {
        require_property(&mut schema, "allow");
    }
    schema
        .as_object_mut()
        .expect("mcp_policy root schema is a JSON object")
        .insert(
            "allOf".to_string(),
            json!([{
                "if": {
                    "properties": { "scope": { "const": "team" } }
                },
                "then": {
                    "required": ["scope_ref"],
                    "properties": { "scope_ref": { "type": "string", "minLength": 1 } }
                }
            }]),
        );
    schema
}

/// Canonical JSON Schema for the `guardrail` resource, derived from the
/// [`Guardrail`](crate::models::Guardrail) struct. `schemars` renders the
/// internally-tagged `GuardrailKind` as a native top-level `oneOf`. Five
/// things need fixing up:
///
/// 1. Unknown-field strictness, which used to live in the config structs as
///    `#[serde(deny_unknown_fields)]` and now lives here. `Guardrail`
///    flattens a tagged enum, so serde hands the whole config subtree to a
///    buffered `Content`: a type-level `deny` there applied to the etcd READ
///    path as well, where it killed every row whose control plane had added
///    a field, and it was invisible in the published schema besides. The
///    write contract is expressed instead by closing every struct-shaped
///    definition and every kind branch — the shared root properties are
///    copied into each branch first, since a closed branch lists only its own
///    kind's fields while the document also carries `name`, `enabled`,
///    `hook_point`, … from the flattened parent. Allowed = root fields ∪
///    `kind` ∪ that kind's fields: exactly what serde enforced, now on the
///    write path only ([`open_unknown_fields`] strips it for the loader).
/// 2. The tagged sub-enums (`KeywordPattern`/`BedrockAWSCredentials`/
///    `BedrockLatencyMode`) lose `deny_unknown_fields` in their `oneOf`
///    branches, so each is re-closed with `additionalProperties: false`.
/// 3. The stringly-typed moderation fields carry closed sets the hand-written
///    schema enforced via `enum`. They stay `String` on the struct (their
///    values flow through `aisix-guardrails` as strings; converting them to
///    Rust enums would churn that crate's processing), so the closed set is
///    injected here into the relevant property.
/// 4. `schemars` leaves discriminator tag fields and collection item schemas
///    without descriptions, so the public schema fills those gaps.
/// 5. `created_at` republishes its `date-time` format (annotation-only).
pub fn guardrail_root_schema() -> Value {
    let mut schema = struct_root_schema::<crate::models::Guardrail>(false);
    let obj = schema
        .as_object_mut()
        .expect("guardrail root schema is a JSON object");

    if let Some(Value::Object(defs)) = obj.get_mut("definitions") {
        for name in [
            "KeywordPattern",
            "BedrockAWSCredentials",
            "BedrockLatencyMode",
        ] {
            if let Some(Value::Array(branches)) =
                defs.get_mut(name).and_then(|d| d.get_mut("oneOf"))
            {
                for branch in branches.iter_mut() {
                    if let Some(b) = branch.as_object_mut() {
                        b.insert("additionalProperties".to_string(), json!(false));
                    }
                }
            }
        }

        set_definition_property_enum(
            defs,
            "PiiDetectorConfig",
            "type",
            json!([
                "email",
                "china_mobile",
                "china_id_card",
                "bank_card",
                "us_ssn",
                "ip_address",
                "api_key",
                "jwt",
                "private_key"
            ]),
        );
        set_definition_property_enum(
            defs,
            "PiiDetectorConfig",
            "action",
            json!(["mask", "block"]),
        );
        set_definition_property_enum(defs, "PiiCustomPattern", "action", json!(["mask", "block"]));
        // Semantic categories rewrite only — the `block` action does not
        // exist for this kind, and the schema says so rather than letting
        // a `block` value be accepted and half-honored (#963).
        set_definition_property_enum(defs, "SmartRedactionCategory", "action", json!(["mask"]));
        // The write path must see candidate patterns spelled out: the
        // serde default is `[]`, and JSON Schema's `minItems` does not
        // apply to an ABSENT property — without `required`, a category
        // with no patterns validates strictly and then fails category
        // compilation, silently skipping the whole row (#963 class).
        if let Some(cat) = defs.get_mut("SmartRedactionCategory") {
            require_property(cat, "candidate_patterns");
        }
        set_definition_property_enum(
            defs,
            "PresidioEntityConfig",
            "action",
            json!(["mask", "block"]),
        );
        set_definition_variant_property_description(
            defs,
            "BedrockAWSCredentials",
            "static",
            "kind",
            "Credential mode for explicitly configured AWS access keys.",
        );
        set_definition_variant_property_description(
            defs,
            "BedrockLatencyMode",
            "serial",
            "kind",
            "Latency mode that waits for the Bedrock guardrail response.",
        );
        set_definition_variant_property_description(
            defs,
            "BedrockLatencyMode",
            "timed",
            "kind",
            "Latency mode that stops waiting after `timeout_ms`.",
        );
        set_definition_variant_property_description(
            defs,
            "KeywordPattern",
            "literal",
            "kind",
            "Pattern type for matching the value as plain text.",
        );
        set_definition_variant_property_description(
            defs,
            "KeywordPattern",
            "regex",
            "kind",
            "Pattern type for matching the value as a regular expression.",
        );
        set_definition_variant_property_description(
            defs,
            "KeywordPattern",
            "literal",
            "value",
            "Literal string to match.",
        );
        set_definition_variant_property_description(
            defs,
            "KeywordPattern",
            "regex",
            "value",
            "Regular expression pattern to match.",
        );
    }

    if let Some(Value::Array(branches)) = obj.get_mut("oneOf") {
        for branch in branches.iter_mut() {
            let Some(b) = branch.as_object_mut() else {
                continue;
            };
            let Some(kind) = branch_kind(b).map(str::to_owned) else {
                continue;
            };
            if let Some(description) = guardrail_kind_description(&kind) {
                set_property_description(b, "kind", description);
            }
            match kind.as_str() {
                "azure_content_safety_text_moderation" => {
                    set_property_enum(
                        b,
                        "output_type",
                        json!(["FourSeverityLevels", "EightSeverityLevels"]),
                    );
                    set_property_enum(
                        b,
                        "text_source",
                        json!(["concatenate_user_content", "concatenate_all_content"]),
                    );
                    set_property_enum(
                        b,
                        "stream_processing_mode",
                        json!(["window", "buffer_full"]),
                    );
                    set_property_items_enum(
                        b,
                        "categories",
                        json!(["Hate", "Sexual", "SelfHarm", "Violence"]),
                    );
                    set_property_items_description(
                        b,
                        "categories",
                        "Azure content category to analyze.",
                    );
                    set_property_items_description(
                        b,
                        "blocklist_names",
                        "Azure blocklist name to match against.",
                    );
                    set_property_additional_properties_description(
                        b,
                        "severity_threshold_by_category",
                        "Severity threshold for the category key.",
                    );
                }
                "aliyun_text_moderation" => {
                    set_property_enum(b, "risk_level_threshold", json!(["low", "medium", "high"]));
                    set_property_enum(
                        b,
                        "stream_processing_mode",
                        json!(["window", "buffer_full"]),
                    );
                }
                "pii" => {
                    set_property_enum(b, "default_action", json!(["mask", "block"]));
                }
                "openai_moderation" => {
                    set_property_additional_properties_description(
                        b,
                        "category_thresholds",
                        "Score threshold for the category key.",
                    );
                }
                "presidio" => {
                    set_property_enum(b, "default_action", json!(["mask", "block"]));
                    set_property_enum(b, "operator", json!(["replace", "mask", "hash", "redact"]));
                }
                "smart_redaction" => {
                    // A semantic row with no categories detects nothing;
                    // the write path must see them spelled out (the serde
                    // default keeps the Rust type read-tolerant), matching
                    // the keyword branch's required `patterns`.
                    match b.get_mut("required").and_then(Value::as_array_mut) {
                        Some(list) => {
                            if !list.iter().any(|v| v.as_str() == Some("categories")) {
                                list.push(json!("categories"));
                            }
                        }
                        None => {
                            b.insert("required".to_string(), json!(["categories"]));
                        }
                    }
                    // This kind rewrites and never blocks, so a runtime
                    // failure has nothing to fail closed INTO —
                    // `fail_open: false` would be accepted-but-inert
                    // (#963). Pre-inserting the pinned property here wins
                    // over the parent copy-in below (`entry().or_insert`),
                    // so the branch keeps the closed value; omission still
                    // gets the default.
                    if let Some(props) = b
                        .entry("properties".to_string())
                        .or_insert_with(|| json!({}))
                        .as_object_mut()
                    {
                        props.insert(
                            "fail_open".to_string(),
                            json!({
                                "type": "boolean",
                                "enum": [true],
                                "default": true,
                                "description": "Semantic guardrails always degrade open: \
                                 an unavailable embedding model releases content \
                                 unmasked and records the degradation, never blocks. \
                                 Only `true` is accepted."
                            }),
                        );
                    }
                }
                "semantic" => {
                    set_property_enum(b, "text_source", json!(["user_messages", "all_messages"]));
                    set_property_enum(b, "on_buffer_exceeded", json!(["fail_closed", "fail_open"]));
                    // Required on the write path, defaulted in the type:
                    // see the field's doc comment. A row saved without it
                    // would screen every request against a model that
                    // resolves to nothing.
                    match b.get_mut("required").and_then(Value::as_array_mut) {
                        Some(list) => {
                            if !list.iter().any(|v| v.as_str() == Some("embedding_model")) {
                                list.push(json!("embedding_model"));
                            }
                        }
                        None => {
                            b.insert("required".to_string(), json!(["embedding_model"]));
                        }
                    }
                }
                _ => {}
            }
        }
    }

    if let Some(created_at) = obj
        .get_mut("properties")
        .and_then(|p| p.get_mut("created_at"))
        .and_then(Value::as_object_mut)
    {
        created_at.insert("format".to_string(), json!("date-time"));
    }

    // Point 1 of the doc comment: the write-path closure the config structs
    // used to carry in their types. Copy the flattened parent's properties
    // into each branch before closing it, or the closure would reject
    // `name`/`enabled`/`hook_point`/… on every document.
    let top_props = obj
        .get("properties")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    if let Some(Value::Array(branches)) = obj.get_mut("oneOf") {
        for branch in branches.iter_mut() {
            let Some(b) = branch.as_object_mut() else {
                continue;
            };
            let props = b
                .entry("properties".to_string())
                .or_insert_with(|| json!({}));
            if let Some(props) = props.as_object_mut() {
                for (key, value) in &top_props {
                    props.entry(key.clone()).or_insert_with(|| value.clone());
                }
            }
            b.insert("additionalProperties".to_string(), json!(false));
        }
    }
    close_definitions(&mut schema);
    schema
}

/// Set a closed `enum` on a oneOf branch's property (for stringly-typed fields
/// whose closed set lives only in the schema, not the Rust type).
fn set_property_enum(branch: &mut serde_json::Map<String, Value>, field: &str, values: Value) {
    if let Some(Value::Object(properties)) = branch.get_mut("properties") {
        set_enum(properties, field, values);
    }
}

fn set_definition_property_enum(
    defs: &mut serde_json::Map<String, Value>,
    definition: &str,
    field: &str,
    values: Value,
) {
    if let Some(Value::Object(properties)) = defs
        .get_mut(definition)
        .and_then(|d| d.get_mut("properties"))
    {
        set_enum(properties, field, values);
    }
}

fn set_definition_variant_property_description(
    defs: &mut serde_json::Map<String, Value>,
    definition: &str,
    variant_kind: &str,
    field: &str,
    description: &str,
) {
    let Some(Value::Array(branches)) = defs.get_mut(definition).and_then(|d| d.get_mut("oneOf"))
    else {
        return;
    };
    for branch in branches {
        let Some(branch) = branch.as_object_mut() else {
            continue;
        };
        if branch_kind(branch) == Some(variant_kind) {
            set_property_description(branch, field, description);
        }
    }
}

fn guardrail_kind_description(kind: &str) -> Option<&'static str> {
    match kind {
        "keyword" => Some("Guardrail provider type for literal and regular expression matching."),
        "bedrock" => Some("Guardrail provider type for Amazon Bedrock Guardrails."),
        "azure_content_safety" => Some("Guardrail provider type for Azure Prompt Shield."),
        "azure_content_safety_text_moderation" => {
            Some("Guardrail provider type for Azure text moderation.")
        }
        "aliyun_text_moderation" => Some("Guardrail provider type for Aliyun text moderation."),
        "pii" => {
            Some("Guardrail provider type for in-process sensitive-data detection and redaction.")
        }
        "lakera" => Some("Guardrail provider type for Lakera Guard screening."),
        "openai_moderation" => Some("Guardrail provider type for the OpenAI Moderation API."),
        "presidio" => {
            Some("Guardrail provider type for self-hosted Microsoft Presidio PII detection and anonymization.")
        }
        "semantic" => Some(
            "Guardrail provider type for embedding-similarity screening against \
             example texts, using an embedding-kind Model.",
        ),
        "smart_redaction" => Some(
            "Guardrail provider type for in-process semantic category detection and redaction using the bundled embedding model.",
        ),
        _ => None,
    }
}

fn set_enum(properties: &mut serde_json::Map<String, Value>, field: &str, values: Value) {
    if let Some(prop) = properties.get_mut(field).and_then(Value::as_object_mut) {
        prop.insert("enum".to_string(), values);
    }
}

fn set_property_description(
    branch: &mut serde_json::Map<String, Value>,
    field: &str,
    description: &str,
) {
    if let Some(prop) = branch
        .get_mut("properties")
        .and_then(|p| p.get_mut(field))
        .and_then(Value::as_object_mut)
    {
        prop.entry("description".to_string())
            .or_insert_with(|| Value::String(description.to_string()));
    }
}

/// Like [`set_property_enum`] but for the `items` of an array property.
fn set_property_items_enum(
    branch: &mut serde_json::Map<String, Value>,
    field: &str,
    values: Value,
) {
    if let Some(items) = branch
        .get_mut("properties")
        .and_then(|p| p.get_mut(field))
        .and_then(|f| f.get_mut("items"))
        .and_then(Value::as_object_mut)
    {
        items.insert("enum".to_string(), values);
    }
}

fn set_property_items_description(
    branch: &mut serde_json::Map<String, Value>,
    field: &str,
    description: &str,
) {
    if let Some(items) = branch
        .get_mut("properties")
        .and_then(|p| p.get_mut(field))
        .and_then(|f| f.get_mut("items"))
        .and_then(Value::as_object_mut)
    {
        items
            .entry("description".to_string())
            .or_insert_with(|| Value::String(description.to_string()));
    }
}

fn set_property_additional_properties_description(
    branch: &mut serde_json::Map<String, Value>,
    field: &str,
    description: &str,
) {
    if let Some(additional_properties) = branch
        .get_mut("properties")
        .and_then(|p| p.get_mut(field))
        .and_then(|f| f.get_mut("additionalProperties"))
        .and_then(Value::as_object_mut)
    {
        additional_properties
            .entry("description".to_string())
            .or_insert_with(|| Value::String(description.to_string()));
    }
}

/// Canonical JSON Schema for the `cache_policy` resource, derived from the
/// [`CachePolicy`](crate::models::CachePolicy) struct. The struct intentionally
/// has no `deny_unknown_fields`, so the schema omits `additionalProperties`
/// (i.e. `true`) — forward-compat fields from a newer cp-api are tolerated.
pub fn cache_policy_root_schema() -> Value {
    struct_root_schema::<crate::models::CachePolicy>(false)
}

/// Canonical JSON Schema for the `observability_exporter` resource, derived
/// from the [`ObservabilityExporter`](crate::models::ObservabilityExporter)
/// struct. `schemars` renders the internally-tagged `ExporterKind` as a native
/// top-level `oneOf`, but two things need fixing up by hand:
///
/// 1. `schemars` drops `deny_unknown_fields` inside tagged-enum branches, and
///    serde does not enforce it there either, so each branch is re-closed with
///    `additionalProperties: false` (rejecting a smuggled plaintext secret).
///    Because a closed branch only lists its own kind's fields, the shared
///    top-level `name`/`enabled` are copied into every branch. This is the
///    WRITE contract: [`open_unknown_fields`] strips it for the loader, which
///    would otherwise lose the whole exporter row — telemetry off for the
///    length of the upgrade window — over one optional field.
/// 2. The `object_store` cloud-identity cross-field rule (cloud_identity ⇒
///    provider ∈ {s3,gcs} and no credential_ref; otherwise credential_ref
///    required) is injected as an `allOf`/`if`/`then`/`else` — `schemars` can't
///    derive cross-field constraints.
///
/// Re-closing each branch also rejects cross-kind field leakage (e.g. a
/// `datadog` exporter carrying an otlp `project`) that the previous
/// single-union-object validator silently accepted; no valid config mixes kinds.
pub fn observability_exporter_root_schema() -> Value {
    let mut schema = struct_root_schema::<crate::models::ObservabilityExporter>(false);
    let obj = schema
        .as_object_mut()
        .expect("observability_exporter root schema is a JSON object");

    let top_props = obj
        .get("properties")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();

    if let Some(Value::Array(branches)) = obj.get_mut("oneOf") {
        for branch in branches.iter_mut() {
            let Some(branch_obj) = branch.as_object_mut() else {
                continue;
            };
            let is_object_store = branch_kind(branch_obj) == Some("object_store");

            let props = branch_obj
                .entry("properties".to_string())
                .or_insert_with(|| json!({}));
            if let Some(props_obj) = props.as_object_mut() {
                for key in ["name", "enabled"] {
                    if let Some(v) = top_props.get(key) {
                        props_obj
                            .entry(key.to_string())
                            .or_insert_with(|| v.clone());
                    }
                }
            }

            if is_object_store {
                branch_obj.insert(
                    "allOf".to_string(),
                    json!([{
                        "if": {
                            "required": ["auth_mode"],
                            "properties": { "auth_mode": { "const": "cloud_identity" } }
                        },
                        "then": { "properties": { "provider": { "enum": ["s3", "gcs"] } } },
                        "else": { "required": ["credential_ref"] }
                    }]),
                );
            }

            branch_obj.insert("additionalProperties".to_string(), json!(false));
        }
    }
    schema
}

/// The `kind` discriminator value of a schemars-generated tagged-enum `oneOf`
/// branch, whether rendered as a `const` or a single-element `enum`.
fn branch_kind(branch: &serde_json::Map<String, Value>) -> Option<&str> {
    let kind = branch.get("properties")?.get("kind")?;
    if let Some(c) = kind.get("const").and_then(Value::as_str) {
        return Some(c);
    }
    kind.get("enum")?.as_array()?.first()?.as_str()
}

/// Canonical JSON Schema for the `rate_limit_policy` resource, derived from the
/// [`RateLimitPolicy`](crate::models::RateLimitPolicy) struct (the `scope`/
/// `window`/dimension/operator closed sets come from their enums) plus the
/// cross-field invariants `schemars` can't express:
///
/// - the classic/conditional form XOR
///   ([`super::rate_limit_policy::rate_limit_policy_form_one_of`]), which also
///   carries the classic form's "at least one of `max_requests`/`max_tokens`";
/// - the `PolicySchedule` day-selector XOR;
/// - closing the `ConditionNode` object variants on the write path: the node
///   is `#[serde(untagged)]`, so serde buffers its content and silently
///   swallows unknown fields inside it, invisible to the write path's serde
///   step and to `serde_ignored` alike — the schema closure is the only
///   guard there, and the only thing [`unknown_field_paths`] can read to
///   report what the opened read schema now lets through (same reasoning as
///   `OnEmbeddingFailure` in [`model_root_schema`]).
///
/// The tree caps (depth/leaf counts), the operator×dimension admission
/// matrix and regex compilability are beyond draft-07 — those live in
/// [`RateLimitPolicy::validate_semantics`], applied by the loader and the
/// file source after parse.
///
/// [`RateLimitPolicy::validate_semantics`]: crate::models::RateLimitPolicy::validate_semantics
pub fn rate_limit_policy_root_schema() -> Value {
    let mut schema = struct_root_schema::<crate::models::RateLimitPolicy>(false);
    let obj = schema
        .as_object_mut()
        .expect("rate_limit_policy root schema is a JSON object");
    obj.insert(
        "oneOf".to_string(),
        super::rate_limit_policy::rate_limit_policy_form_one_of(),
    );
    let defs = obj
        .get_mut("definitions")
        .and_then(Value::as_object_mut)
        .expect("rate_limit_policy schema has definitions");
    // The schedule day-selector XOR is the same kind of cross-field
    // invariant, one level down in the definitions.
    defs.get_mut("PolicySchedule")
        .and_then(Value::as_object_mut)
        .expect("rate_limit_policy schema defines PolicySchedule")
        .insert(
            "oneOf".to_string(),
            super::rate_limit_policy::policy_schedule_one_of(),
        );
    for def in ["PolicyCondition", "ConditionGroup"] {
        defs.get_mut(def)
            .and_then(Value::as_object_mut)
            .unwrap_or_else(|| panic!("rate_limit_policy schema defines {def}"))
            .insert("additionalProperties".to_string(), json!(false));
    }
    schema
}

/// Canonical JSON Schema for the `guardrail_attachment` resource, derived from
/// the [`GuardrailAttachment`](crate::models::GuardrailAttachment) struct. Uses
/// the nullable `Option` representation (`scope_id` is `null` for `env`-scoped
/// attachments) and stays open (no `deny_unknown_fields`): cp-api includes an
/// `env_id` the DP ignores.
pub fn guardrail_attachment_root_schema() -> Value {
    struct_root_schema::<crate::models::GuardrailAttachment>(true)
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
    fn model_ensemble_form_passes() {
        let v = json!({
            "display_name": "council",
            "ensemble": {
                "panel": [
                    {"model": "my-gpt4", "temperature": 0.5},
                    {"model": "my-claude", "temperature": 1.0}
                ],
                "judge": {"model": "my-opus"},
                "min_responses": 2,
                "timeout_ms": 45000
            }
        });
        validate_model(&v).unwrap();
    }

    #[test]
    fn model_ensemble_can_be_ip_restricted_and_rate_limited() {
        // Top-level gates apply to the ensemble entry model too.
        let v = json!({
            "display_name": "council",
            "ensemble": {
                "panel": [{"model": "a"}, {"model": "b"}],
                "judge": {"model": "j"}
            },
            "allowed_cidrs": ["10.0.0.0/8"],
            "rate_limit": {"rpm": 60}
        });
        validate_model(&v).unwrap();
    }

    #[test]
    fn model_rate_limit_accepts_all_request_windows_incl_rps_rph() {
        // Regression for #644: the inline rate_limit schema is derived from the
        // RateLimit struct (#638), so every request-count window — rps/rpm/rph/
        // rpd — alongside the token windows and concurrency must be accepted,
        // not just rpm/rpd/tpm/tpd/concurrency.
        let v = json!({
            "display_name": "my-gpt4",
            "provider": "openai",
            "model_name": "gpt-4o",
            "provider_key_id": "11111111-1111-1111-1111-111111111111",
            "rate_limit": {
                "rps": 10, "rpm": 100, "rph": 1000, "rpd": 10000,
                "tpm": 100000, "tpd": 1000000, "concurrency": 5
            }
        });
        validate_model(&v).unwrap();
    }

    #[test]
    fn model_ensemble_with_direct_fields_fails() {
        // ensemble is mutually exclusive with the direct upstream triple.
        let v = json!({
            "display_name": "x",
            "provider": "openai",
            "model_name": "gpt-4o",
            "provider_key_id": "pk-1",
            "ensemble": {
                "panel": [{"model": "a"}],
                "judge": {"model": "j"}
            }
        });
        assert!(validate_model(&v).is_err());
    }

    #[test]
    fn model_ensemble_with_routing_fails() {
        // A model can't be both an ensemble and a router.
        let v = json!({
            "display_name": "x",
            "routing": {"targets": [{"model": "a"}]},
            "ensemble": {
                "panel": [{"model": "a"}],
                "judge": {"model": "j"}
            }
        });
        assert!(validate_model(&v).is_err());
    }

    #[test]
    fn model_ensemble_missing_judge_fails() {
        let v = json!({
            "display_name": "x",
            "ensemble": {
                "panel": [{"model": "a"}, {"model": "b"}]
            }
        });
        assert!(validate_model(&v).is_err());
    }

    #[test]
    fn model_ensemble_empty_panel_fails() {
        let v = json!({
            "display_name": "x",
            "ensemble": {
                "panel": [],
                "judge": {"model": "j"}
            }
        });
        assert!(validate_model(&v).is_err());
    }

    #[test]
    fn model_ensemble_unknown_panel_field_fails() {
        let v = json!({
            "display_name": "x",
            "ensemble": {
                "panel": [{"model": "a", "bogus": true}],
                "judge": {"model": "j"}
            }
        });
        assert!(validate_model(&v).is_err());
    }

    // ---- semantic-routing + embedding-modality schema tests (#641) ----

    #[test]
    fn model_semantic_form_passes() {
        let v = json!({
            "display_name": "prod-chat",
            "semantic": {
                "embedding_model": "bge-m3",
                "routes": [
                    {
                        "name": "legal",
                        "target": "claude-opus",
                        "description": "Contract & legal risk analysis",
                        "examples": ["分析这份合同里的潜在风险", "Review this NDA"],
                        "threshold": 0.8
                    },
                    {"name": "translate", "target": "gpt-4o-mini", "examples": ["帮我翻译这句话"]}
                ],
                "default": "gpt-4o",
                "match": {"distance_metric": "cosine", "aggregation": "max", "threshold": 0.75},
                "embedding_timeout_ms": 500,
                "on_embedding_failure": {"target": "gpt-4o-mini"}
            }
        });
        validate_model(&v).unwrap();
    }

    #[test]
    fn model_semantic_minimal_form_passes() {
        let v = json!({
            "display_name": "prod-chat",
            "semantic": {
                "embedding_model": "bge-m3",
                "routes": [{"name": "a", "target": "m", "examples": ["hi"]}],
                "default": "gpt-4o",
                "match": {"threshold": 0.5}
            }
        });
        validate_model(&v).unwrap();
    }

    #[test]
    fn model_semantic_can_be_ip_restricted_and_rate_limited() {
        // Top-level gates apply to the semantic router entry too.
        let v = json!({
            "display_name": "prod-chat",
            "semantic": {
                "embedding_model": "e",
                "routes": [{"name": "a", "target": "m", "examples": ["x"]}],
                "default": "d",
                "match": {"threshold": 0.5}
            },
            "allowed_cidrs": ["10.0.0.0/8"],
            "rate_limit": {"rpm": 60}
        });
        validate_model(&v).unwrap();
    }

    #[test]
    fn model_semantic_on_embedding_failure_accepts_bare_modes() {
        for mode in ["default", "fail"] {
            let v = json!({
                "display_name": "prod-chat",
                "semantic": {
                    "embedding_model": "e",
                    "routes": [{"name": "a", "target": "m", "examples": ["x"]}],
                    "default": "d",
                    "match": {"threshold": 0.5},
                    "on_embedding_failure": mode
                }
            });
            validate_model(&v).unwrap_or_else(|e| panic!("mode {mode:?} must validate: {e:?}"));
        }
    }

    #[test]
    fn model_semantic_with_direct_fields_fails() {
        // semantic is mutually exclusive with the direct upstream triple.
        let v = json!({
            "display_name": "x",
            "provider": "openai",
            "model_name": "gpt-4o",
            "provider_key_id": "pk-1",
            "semantic": {
                "embedding_model": "e",
                "routes": [{"name": "a", "target": "m", "examples": ["x"]}],
                "default": "d",
                "match": {"threshold": 0.5}
            }
        });
        assert!(validate_model(&v).is_err());
    }

    #[test]
    fn model_semantic_with_routing_fails() {
        let v = json!({
            "display_name": "x",
            "routing": {"targets": [{"model": "a"}]},
            "semantic": {
                "embedding_model": "e",
                "routes": [{"name": "a", "target": "m", "examples": ["x"]}],
                "default": "d",
                "match": {"threshold": 0.5}
            }
        });
        assert!(validate_model(&v).is_err());
    }

    #[test]
    fn model_semantic_missing_required_fields_fails() {
        // Missing `default`.
        let v = json!({
            "display_name": "x",
            "semantic": {
                "embedding_model": "e",
                "routes": [{"name": "a", "target": "m", "examples": ["x"]}],
                "match": {"threshold": 0.5}
            }
        });
        assert!(validate_model(&v).is_err());
    }

    #[test]
    fn model_semantic_empty_routes_fails() {
        let v = json!({
            "display_name": "x",
            "semantic": {
                "embedding_model": "e",
                "routes": [],
                "default": "d",
                "match": {"threshold": 0.5}
            }
        });
        assert!(validate_model(&v).is_err());
    }

    #[test]
    fn model_semantic_route_without_examples_fails() {
        // examples-only matching: a route needs at least one example.
        let v = json!({
            "display_name": "x",
            "semantic": {
                "embedding_model": "e",
                "routes": [{"name": "a", "target": "m", "examples": []}],
                "default": "d",
                "match": {"threshold": 0.5}
            }
        });
        assert!(validate_model(&v).is_err());
    }

    #[test]
    fn model_semantic_threshold_out_of_range_fails() {
        let v = json!({
            "display_name": "x",
            "semantic": {
                "embedding_model": "e",
                "routes": [{"name": "a", "target": "m", "examples": ["x"]}],
                "default": "d",
                "match": {"threshold": 1.5}
            }
        });
        assert!(validate_model(&v).is_err());
    }

    #[test]
    fn model_embedding_modality_on_direct_passes() {
        // An embedding model is a direct model that also carries the
        // embedding-modality block.
        let v = json!({
            "display_name": "bge-m3",
            "provider": "openai",
            "model_name": "bge-m3",
            "provider_key_id": "pk-1",
            "embedding": {"dimensions": 1024, "normalize": false}
        });
        validate_model(&v).unwrap();
    }

    #[test]
    fn model_embedding_without_dimensions_fails() {
        let v = json!({
            "display_name": "bge-m3",
            "provider": "openai",
            "model_name": "bge-m3",
            "provider_key_id": "pk-1",
            "embedding": {"normalize": true}
        });
        assert!(validate_model(&v).is_err());
    }

    #[test]
    fn model_embedding_on_routing_fails() {
        // The embedding block is modality metadata on a direct model — it
        // has no meaning on a virtual router.
        let v = json!({
            "display_name": "x",
            "routing": {"targets": [{"model": "a"}]},
            "embedding": {"dimensions": 1024}
        });
        assert!(validate_model(&v).is_err());
    }

    #[test]
    fn model_embedding_on_semantic_fails() {
        let v = json!({
            "display_name": "x",
            "semantic": {
                "embedding_model": "e",
                "routes": [{"name": "a", "target": "m", "examples": ["x"]}],
                "default": "d",
                "match": {"threshold": 0.5}
            },
            "embedding": {"dimensions": 1024}
        });
        assert!(validate_model(&v).is_err());
    }

    #[test]
    fn model_allowed_cidrs_passes_on_direct_and_routing() {
        // Direct model with an IP allowlist (#557).
        let direct = json!({
            "display_name": "ip-restricted",
            "provider": "openai",
            "model_name": "gpt-4o",
            "provider_key_id": "11111111-1111-1111-1111-111111111111",
            "allowed_cidrs": ["10.0.0.0/8", "2001:db8::/32"]
        });
        validate_model(&direct).unwrap();

        // Routing (Model Group) model can also be IP-restricted — the gate
        // binds to the requested model name regardless of its shape.
        let routing = json!({
            "display_name": "router-restricted",
            "routing": {
                "strategy": "failover",
                "targets": [{"model": "my-gpt4"}]
            },
            "allowed_cidrs": ["10.0.0.0/8"]
        });
        validate_model(&routing).unwrap();
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
    fn apikey_with_team_and_user_fields_passes() {
        let v = json!({
            "key_hash":"9df37f5e7cbc3c391d872742b5f286c242e733a09add9eeaa4d26a599bd90b20",
            "allowed_models":["gpt-4o"],
            "team_id": "team-uuid-1",
            "user_id": "member-uuid-1"
        });
        validate_apikey(&v).unwrap();
    }

    #[test]
    fn apikey_with_null_team_and_user_fields_passes() {
        let v = json!({
            "key_hash":"9df37f5e7cbc3c391d872742b5f286c242e733a09add9eeaa4d26a599bd90b20",
            "allowed_models":["gpt-4o"],
            "team_id": null,
            "user_id": null
        });
        validate_apikey(&v).unwrap();
    }

    #[test]
    fn apikey_rate_limit_accepts_rps_and_rph() {
        // Regression for #644: inline rate_limit on a caller API key must accept
        // the per-second and per-hour request windows too, not only
        // rpm/rpd/tpm/tpd/concurrency.
        let v = json!({
            "key_hash":"9df37f5e7cbc3c391d872742b5f286c242e733a09add9eeaa4d26a599bd90b20",
            "allowed_models":["gpt-4o"],
            "rate_limit": {"rps": 5, "rph": 500}
        });
        validate_apikey(&v).unwrap();
    }

    #[test]
    fn apikey_mcp_access_block_passes() {
        let v = json!({
            "key_hash":"9df37f5e7cbc3c391d872742b5f286c242e733a09add9eeaa4d26a599bd90b20",
            "allowed_models":["gpt-4o"],
            "mcp_access": {"allow": ["*"]}
        });
        validate_apikey(&v).unwrap();
        let v = json!({
            "key_hash":"9df37f5e7cbc3c391d872742b5f286c242e733a09add9eeaa4d26a599bd90b20",
            "allowed_models":["gpt-4o"],
            "mcp_access": {"allow": ["github__*"], "deny": ["github__delete_repo"]}
        });
        validate_apikey(&v).unwrap();
    }

    #[test]
    fn apikey_mcp_access_requires_an_explicit_allow_side() {
        // A block carrying only `deny` would silently grant nothing; the
        // schema forces the author to say what the key allows.
        let v = json!({
            "key_hash":"9df37f5e7cbc3c391d872742b5f286c242e733a09add9eeaa4d26a599bd90b20",
            "allowed_models":[],
            "mcp_access": {"deny": ["github__*"]}
        });
        assert!(validate_apikey(&v).is_err());
    }

    #[test]
    fn apikey_mcp_access_rejects_the_removed_mode_field() {
        // Write path: `mode` is gone from the authored shape. Read path:
        // the CP projects `"mode": "deny"` as a tombstone so 0.9.x DPs
        // load the row fail-closed, and this generation's loader must
        // keep accepting the hybrid document.
        let v = json!({
            "key_hash":"9df37f5e7cbc3c391d872742b5f286c242e733a09add9eeaa4d26a599bd90b20",
            "allowed_models":[],
            "mcp_access": {"mode": "inherit", "allow": ["*"]}
        });
        assert!(validate_apikey(&v).is_err());
        validate_apikey_lenient(&v).unwrap();
        validate_apikey_lenient(&json!({
            "key_hash":"9df37f5e7cbc3c391d872742b5f286c242e733a09add9eeaa4d26a599bd90b20",
            "allowed_models":[],
            "mcp_access": {"mode": "deny", "allow": ["github__*"], "deny": ["x__y"]}
        }))
        .unwrap();
    }

    #[test]
    fn apikey_rejects_the_removed_allowed_tools_field() {
        let v = json!({
            "key_hash":"9df37f5e7cbc3c391d872742b5f286c242e733a09add9eeaa4d26a599bd90b20",
            "allowed_models":[],
            "allowed_tools": ["github__*"]
        });
        assert!(validate_apikey(&v).is_err());
    }

    #[test]
    fn mcp_policy_env_and_team_forms_pass() {
        validate_mcp_policy(&json!({
            "scope": "env",
            "allow": ["github__*"],
            "deny": ["github__delete_repo"]
        }))
        .unwrap();
        validate_mcp_policy(&json!({
            "scope": "team",
            "scope_ref": "team-uuid-1",
            "allow": ["*"],
            "enabled": true
        }))
        .unwrap();
    }

    #[test]
    fn mcp_policy_team_scope_requires_scope_ref() {
        // A team row without its team id could shadow the environment
        // layer; the cross-field guard rejects it at the schema gate.
        assert!(validate_mcp_policy(&json!({"scope": "team", "allow": ["*"]})).is_err());
        assert!(
            validate_mcp_policy(&json!({"scope": "team", "scope_ref": null, "allow": ["*"]}))
                .is_err()
        );
        // The environment layer carries no scope_ref.
        validate_mcp_policy(&json!({"scope": "env", "allow": []})).unwrap();
    }

    #[test]
    fn the_lenient_schema_still_loads_a_pre_layer_row() {
        // Documents projected before the layered shape carry a `mode`
        // and no `allow`. The strict write path rejects them, but the
        // runtime loader must still accept them: a rejected `api_key`
        // row is skipped entirely, so the key would stop authenticating
        // for ALL traffic rather than merely losing MCP access.
        let key = json!({
            "key_hash":"9df37f5e7cbc3c391d872742b5f286c242e733a09add9eeaa4d26a599bd90b20",
            "allowed_models":[],
            "allowed_tools":["github__*"],
            "mcp_access": {"mode": "inherit"}
        });
        validate_apikey_lenient(&key).unwrap();
        assert!(validate_apikey(&key).is_err());

        let policy = json!({"scope": "env", "mode": "all"});
        validate_mcp_policy_lenient(&policy).unwrap();
        assert!(validate_mcp_policy(&policy).is_err());
    }

    #[test]
    fn mcp_policy_requires_an_explicit_allow_side() {
        assert!(validate_mcp_policy(&json!({"scope": "env"})).is_err());
        assert!(validate_mcp_policy(&json!({"scope": "env", "deny": ["github__*"]})).is_err());
    }

    #[test]
    fn mcp_policy_rejects_unknown_fields_and_values() {
        assert!(validate_mcp_policy(&json!({"scope": "org", "allow": ["*"]})).is_err());
        assert!(validate_mcp_policy(&json!({"scope": "env", "allow": ["*"], "rogue": 1})).is_err());
        // `mode` is gone; a payload still carrying it is a write from a
        // control plane that has not caught up.
        assert!(
            validate_mcp_policy(&json!({"scope": "env", "mode": "all", "allow": ["*"]})).is_err()
        );
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
    fn routing_fallback_on_statuses_range_is_enforced() {
        // AISIX-Cloud#1012: entries outside 400-599 are rejected by the
        // same committed-schema validation the admin API and etcd watch
        // paths share; an in-range list passes.
        let bad = json!({
            "display_name": "router-fos",
            "routing": {
                "targets": [{"model": "x"}],
                "fallback_on_statuses": [300]
            }
        });
        assert!(validate_model(&bad).is_err());
        let good = json!({
            "display_name": "router-fos",
            "routing": {
                "targets": [{"model": "x"}],
                "fallback_on_statuses": [408, 422]
            }
        });
        validate_model(&good).unwrap();
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
    fn routing_when_all_unavailable_fail_passes() {
        let v = json!({
            "display_name": "router-1",
            "routing": {
                "targets": [{"model": "a"}],
                "when_all_unavailable": "fail"
            }
        });
        validate_model(&v).unwrap();
    }

    #[test]
    fn routing_when_all_unavailable_try_anyway_passes() {
        let v = json!({
            "display_name": "router-1",
            "routing": {
                "targets": [{"model": "a"}],
                "when_all_unavailable": "try_anyway"
            }
        });
        validate_model(&v).unwrap();
    }

    #[test]
    fn routing_when_all_unavailable_rejects_unknown_value() {
        let v = json!({
            "display_name": "router-1",
            "routing": {
                "targets": [{"model": "a"}],
                "when_all_unavailable": "yolo"
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

    #[test]
    fn guardrail_azure_content_safety_passes() {
        // Regression for #437: the loader JSON schema must accept the
        // azure_content_safety kind, not just the Rust struct. timeout_ms
        // omitted here — it's optional (defaults to 5000 on the struct).
        let v = json!({
            "name": "prompt-shield",
            "kind": "azure_content_safety",
            "hook_point": "input",
            "endpoint": "https://my-resource.cognitiveservices.azure.com",
            "api_key": "plaintext-key"
        });
        validate_guardrail(&v).unwrap();
    }

    #[test]
    fn guardrail_azure_content_safety_with_timeout_passes() {
        let v = json!({
            "name": "prompt-shield",
            "kind": "azure_content_safety",
            "endpoint": "https://r.cognitiveservices.azure.com",
            "api_key": "k",
            "timeout_ms": 3000
        });
        validate_guardrail(&v).unwrap();
    }

    #[test]
    fn guardrail_azure_content_safety_missing_api_key_rejected() {
        let v = json!({
            "name": "g",
            "kind": "azure_content_safety",
            "endpoint": "https://r.cognitiveservices.azure.com"
        });
        assert!(validate_guardrail(&v).is_err());
    }

    #[test]
    fn guardrail_azure_content_safety_max_timeout_passes() {
        // Guards the exact regression class of #437: the loader schema
        // must accept everything AzureContentSafetyConfig's timeout_ms
        // (u32) accepts, INCLUDING u32::MAX. A future edit that tightens
        // the schema below u32::MAX would make the loader stricter than
        // the struct and silently drop valid rows — this test fails loud.
        let v = json!({
            "name": "g",
            "kind": "azure_content_safety",
            "endpoint": "https://r.cognitiveservices.azure.com",
            "api_key": "k",
            "timeout_ms": 4_294_967_295u64
        });
        validate_guardrail(&v).unwrap();
    }

    #[test]
    fn guardrail_azure_content_safety_timeout_overflow_rejected() {
        // u32::MAX + 1 — beyond what the struct can deserialize. The
        // schema must reject it at the gate so the loader skips the row
        // cleanly instead of surfacing an opaque serde error downstream.
        let v = json!({
            "name": "g",
            "kind": "azure_content_safety",
            "endpoint": "https://r.cognitiveservices.azure.com",
            "api_key": "k",
            "timeout_ms": 4_294_967_296u64
        });
        assert!(validate_guardrail(&v).is_err());
    }

    #[test]
    fn guardrail_aliyun_text_moderation_passes() {
        // Minimal row: region + access keys. Optional fields (endpoint,
        // threshold, streaming params) omitted — the struct applies defaults.
        let v = json!({
            "name": "aliyun-guard",
            "kind": "aliyun_text_moderation",
            "hook_point": "both",
            "region": "cn-shanghai",
            "access_key_id": "LTAI_EXAMPLE",
            "access_key_secret": "plaintext-secret"
        });
        validate_guardrail(&v).unwrap();
    }

    #[test]
    fn guardrail_aliyun_text_moderation_with_optional_fields_passes() {
        let v = json!({
            "name": "aliyun-guard",
            "kind": "aliyun_text_moderation",
            "region": "cn-beijing",
            "endpoint": "http://127.0.0.1:8080",
            "access_key_id": "id",
            "access_key_secret": "secret",
            "risk_level_threshold": "medium",
            "timeout_ms": 3000,
            "stream_processing_mode": "buffer_full"
        });
        validate_guardrail(&v).unwrap();
    }

    #[test]
    fn guardrail_aliyun_text_moderation_missing_secret_rejected() {
        let v = json!({
            "name": "g",
            "kind": "aliyun_text_moderation",
            "region": "cn-shanghai",
            "access_key_id": "id"
        });
        assert!(validate_guardrail(&v).is_err());
    }

    #[test]
    fn guardrail_aliyun_text_moderation_bad_threshold_rejected() {
        let v = json!({
            "name": "g",
            "kind": "aliyun_text_moderation",
            "region": "cn-shanghai",
            "access_key_id": "id",
            "access_key_secret": "s",
            "risk_level_threshold": "none"
        });
        assert!(validate_guardrail(&v).is_err());
    }

    #[test]
    fn guardrail_rejects_invalid_string_enums() {
        let cases = [
            json!({
                "name": "g",
                "kind": "azure_content_safety_text_moderation",
                "endpoint": "https://example.cognitiveservices.azure.com",
                "api_key": "key",
                "output_type": "TwoSeverityLevels"
            }),
            json!({
                "name": "g",
                "kind": "azure_content_safety_text_moderation",
                "endpoint": "https://example.cognitiveservices.azure.com",
                "api_key": "key",
                "categories": ["Hate", "Spam"]
            }),
            json!({
                "name": "g",
                "kind": "azure_content_safety_text_moderation",
                "endpoint": "https://example.cognitiveservices.azure.com",
                "api_key": "key",
                "text_source": "assistant_only"
            }),
            json!({
                "name": "g",
                "kind": "azure_content_safety_text_moderation",
                "endpoint": "https://example.cognitiveservices.azure.com",
                "api_key": "key",
                "stream_processing_mode": "chunked"
            }),
            json!({
                "name": "g",
                "kind": "pii",
                "detectors": [{ "type": "email", "action": "redact" }]
            }),
            json!({
                "name": "g",
                "kind": "pii",
                "custom_patterns": [{
                    "name": "employee_id",
                    "regex": "\\bEMP-\\d+\\b",
                    "action": "redact"
                }]
            }),
            json!({
                "name": "g",
                "kind": "pii",
                "default_action": "redact",
                "detectors": [{ "type": "email" }]
            }),
            json!({
                "name": "g",
                "kind": "pii",
                "detectors": [{ "type": "driver_license" }]
            }),
            json!({
                "name": "g",
                "kind": "presidio",
                "analyzer_url": "http://presidio-analyzer:3000",
                "anonymizer_url": "http://presidio-anonymizer:3000",
                "default_action": "redact"
            }),
            json!({
                "name": "g",
                "kind": "presidio",
                "analyzer_url": "http://presidio-analyzer:3000",
                "anonymizer_url": "http://presidio-anonymizer:3000",
                "entities": [{ "type": "EMAIL_ADDRESS", "action": "redact" }]
            }),
            json!({
                "name": "g",
                "kind": "presidio",
                "analyzer_url": "http://presidio-analyzer:3000",
                "anonymizer_url": "http://presidio-anonymizer:3000",
                "operator": "tokenize"
            }),
        ];

        for value in cases {
            assert!(
                validate_guardrail(&value).is_err(),
                "guardrail schema accepted invalid enum value: {value}",
            );
        }
    }

    #[test]
    fn guardrail_fail_safe_fields_stay_schema_open() {
        let cases = [
            json!({
                "name": "g",
                "kind": "keyword",
                "patterns": [],
                "enforcement_mode": "audit",
                "direction": "sideways"
            }),
            json!({
                "name": "g",
                "kind": "azure_content_safety_text_moderation",
                "endpoint": "https://example.cognitiveservices.azure.com",
                "api_key": "key",
                "on_buffer_exceeded": "drop"
            }),
            json!({
                "name": "g",
                "kind": "aliyun_text_moderation",
                "region": "cn-shanghai",
                "access_key_id": "id",
                "access_key_secret": "s",
                "on_buffer_exceeded": "drop"
            }),
            json!({
                "name": "g",
                "kind": "pii",
                "detectors": [{ "type": "email" }],
                "on_buffer_exceeded": "drop"
            }),
            json!({
                "name": "g",
                "kind": "lakera",
                "api_key": "key",
                "on_buffer_exceeded": "drop"
            }),
            json!({
                "name": "g",
                "kind": "presidio",
                "analyzer_url": "http://presidio-analyzer:3000",
                "anonymizer_url": "http://presidio-anonymizer:3000",
                "on_buffer_exceeded": "drop"
            }),
        ];

        for value in cases {
            validate_guardrail(&value).unwrap();
        }
    }

    #[test]
    fn guardrail_openai_moderation_model_stays_open() {
        let v = json!({
            "name": "openai-mod",
            "kind": "openai_moderation",
            "api_key": "plaintext-key",
            "model": "future-moderation-model"
        });
        validate_guardrail(&v).unwrap();
    }

    // ---- observability_exporter schema tests ----

    #[test]
    fn exporter_otlp_http_happy_path() {
        let v = json!({
            "name": "honeycomb",
            "kind": "otlp_http",
            "endpoint": "https://api.honeycomb.io/v1/traces",
            "headers": { "x-honeycomb-team": "abc" }
        });
        validate_observability_exporter(&v).unwrap();
    }

    #[test]
    fn exporter_otlp_http_rejects_plain_http_endpoint() {
        let v = json!({
            "name": "x",
            "kind": "otlp_http",
            "endpoint": "http://api.honeycomb.io/v1/traces"
        });
        assert!(validate_observability_exporter(&v).is_err());
    }

    #[test]
    fn exporter_otlp_http_accepts_in_range_knobs() {
        // #519 B.2: sampling + content capture are real per-exporter knobs.
        for rate in [0.0, 0.5, 1.0] {
            let v = json!({
                "name": "otlp-knobs",
                "kind": "otlp_http",
                "endpoint": "https://api.honeycomb.io/v1/traces",
                "sample_rate": rate,
                "content_mode": "full",
                "content_max_bytes": 4096
            });
            validate_observability_exporter(&v).unwrap();
        }
    }

    #[test]
    fn exporter_otlp_http_rejects_out_of_range_sample_rate() {
        for rate in [-0.1, 1.1, 2.0] {
            let v = json!({
                "name": "x",
                "kind": "otlp_http",
                "endpoint": "https://api.honeycomb.io/v1/traces",
                "sample_rate": rate
            });
            assert!(
                validate_observability_exporter(&v).is_err(),
                "sample_rate {rate} must be rejected"
            );
        }
    }

    #[test]
    fn exporter_aliyun_sls_happy_path() {
        let v = json!({
            "name": "sls-prod",
            "kind": "aliyun_sls",
            "endpoint": "ap-southeast-3.log.aliyuncs.com",
            "project": "aisix-obs",
            "logstore": "request-events",
            "credential_ref": "sls-prod"
        });
        validate_observability_exporter(&v).unwrap();
    }

    #[test]
    fn exporter_aliyun_sls_allows_loopback_mock_endpoint() {
        // The L2 e2e points the DP at a local mock SLS over http://.
        let v = json!({
            "name": "sls-e2e",
            "kind": "aliyun_sls",
            "endpoint": "http://mock-sls:9000",
            "project": "p",
            "logstore": "l",
            "credential_ref": "mock"
        });
        validate_observability_exporter(&v).unwrap();
    }

    #[test]
    fn exporter_object_store_happy_path() {
        let v = json!({
            "name": "acme-s3",
            "kind": "object_store",
            "provider": "s3",
            "bucket": "acme-aisix-events",
            "prefix": "ai-gateway",
            "region": "us-east-1",
            "credential_ref": "acme-s3"
        });
        validate_observability_exporter(&v).unwrap();
    }

    #[test]
    fn exporter_object_store_requires_core_fields() {
        // Each config missing one required object_store field is rejected.
        let cases = [
            json!({"name":"x","kind":"object_store","bucket":"b","prefix":"p","credential_ref":"r"}),
            json!({"name":"x","kind":"object_store","provider":"s3","prefix":"p","credential_ref":"r"}),
            json!({"name":"x","kind":"object_store","provider":"s3","bucket":"b","credential_ref":"r"}),
            json!({"name":"x","kind":"object_store","provider":"s3","bucket":"b","prefix":"p"}),
        ];
        for v in cases {
            assert!(
                validate_observability_exporter(&v).is_err(),
                "incomplete object_store config must be rejected: {v}"
            );
        }
    }

    #[test]
    fn exporter_object_store_rejects_bad_provider() {
        let v = json!({
            "name": "x", "kind": "object_store",
            "provider": "wasabi", "bucket": "b", "prefix": "p", "credential_ref": "r"
        });
        assert!(validate_observability_exporter(&v).is_err());
    }

    #[test]
    fn exporter_object_store_cloud_identity_omits_credential_ref() {
        // cloud_identity (S3 / GCS): the DP uses its own attached identity, so
        // credential_ref is NOT required.
        for provider in ["s3", "gcs"] {
            let v = json!({
                "name": "x", "kind": "object_store",
                "provider": provider, "bucket": "b", "prefix": "p",
                "auth_mode": "cloud_identity"
            });
            validate_observability_exporter(&v)
                .unwrap_or_else(|e| panic!("cloud_identity {provider} should validate: {e:?}"));
        }
    }

    #[test]
    fn exporter_object_store_cloud_identity_rejects_azure() {
        // Azure cloud_identity is unsupported (managed identity needs a
        // non-secret account name the keyless config does not carry).
        let v = json!({
            "name": "x", "kind": "object_store",
            "provider": "azure_blob", "bucket": "c", "prefix": "p",
            "auth_mode": "cloud_identity"
        });
        assert!(validate_observability_exporter(&v).is_err());
    }

    #[test]
    fn exporter_object_store_credential_ref_mode_still_requires_ref() {
        // Default (no auth_mode) and explicit credential_ref both require the
        // ref — only cloud_identity drops it.
        for v in [
            json!({"name":"x","kind":"object_store","provider":"s3","bucket":"b","prefix":"p"}),
            json!({"name":"x","kind":"object_store","provider":"s3","bucket":"b","prefix":"p","auth_mode":"credential_ref"}),
        ] {
            assert!(
                validate_observability_exporter(&v).is_err(),
                "credential_ref must be required outside cloud_identity: {v}"
            );
        }
    }

    #[test]
    fn exporter_object_store_allows_loopback_minio_endpoint() {
        // The e2e points the S3 sink at a local MinIO over http://.
        let v = json!({
            "name": "s3-e2e", "kind": "object_store",
            "provider": "s3", "bucket": "b", "prefix": "p",
            "endpoint": "http://minio:9000", "credential_ref": "mock"
        });
        validate_observability_exporter(&v).unwrap();
    }

    #[test]
    fn exporter_object_store_rejects_plaintext_non_loopback_endpoint() {
        // A non-loopback plaintext endpoint must be rejected — no exfil to an
        // arbitrary http host.
        let v = json!({
            "name": "x", "kind": "object_store",
            "provider": "s3", "bucket": "b", "prefix": "p",
            "endpoint": "http://evil.example.com", "credential_ref": "r"
        });
        assert!(validate_observability_exporter(&v).is_err());
    }

    #[test]
    fn exporter_aliyun_sls_requires_project_logstore_credential() {
        for missing in ["project", "logstore", "credential_ref"] {
            let mut v = json!({
                "name": "x",
                "kind": "aliyun_sls",
                "endpoint": "ap-southeast-3.log.aliyuncs.com",
                "project": "p",
                "logstore": "l",
                "credential_ref": "r"
            });
            v.as_object_mut().unwrap().remove(missing);
            assert!(
                validate_observability_exporter(&v).is_err(),
                "missing `{missing}` must be rejected"
            );
        }
    }

    #[test]
    fn exporter_aliyun_sls_rejects_plaintext_credentials() {
        // No AccessKey field is allowed at the schema layer either —
        // `additionalProperties: false` rejects it before serde runs.
        let v = json!({
            "name": "x",
            "kind": "aliyun_sls",
            "endpoint": "ap-southeast-3.log.aliyuncs.com",
            "project": "p",
            "logstore": "l",
            "credential_ref": "r",
            "access_key_secret": "AKIASECRET"
        });
        assert!(validate_observability_exporter(&v).is_err());
    }

    #[test]
    fn exporter_aliyun_sls_content_capture_fields() {
        let base = |extra: serde_json::Value| {
            let mut v = json!({
                "name": "x",
                "kind": "aliyun_sls",
                "endpoint": "ap-southeast-3.log.aliyuncs.com",
                "project": "p",
                "logstore": "l",
                "credential_ref": "r"
            });
            let obj = v.as_object_mut().unwrap();
            for (k, val) in extra.as_object().unwrap() {
                obj.insert(k.clone(), val.clone());
            }
            v
        };
        // Opt-in content capture validates.
        validate_observability_exporter(&base(
            json!({ "content_mode": "full", "content_max_bytes": 4096 }),
        ))
        .unwrap();
        // Unknown content_mode is rejected.
        assert!(
            validate_observability_exporter(&base(json!({ "content_mode": "verbose" }))).is_err()
        );
        // content_max_bytes must be a positive integer.
        assert!(validate_observability_exporter(&base(json!({ "content_max_bytes": 0 }))).is_err());
    }

    #[test]
    fn exporter_rejects_unknown_kind() {
        let v = json!({ "name": "x", "kind": "splunk_hec", "endpoint": "https://x" });
        assert!(validate_observability_exporter(&v).is_err());
    }

    #[test]
    fn exporter_datadog_happy_path() {
        let v = json!({
            "name": "datadog-prod",
            "kind": "datadog",
            "site": "datadoghq.com",
            "credential_ref": "datadog-prod",
            "service": "ai-gateway",
            "ddsource": "aisix-ai-gateway",
            "tags": ["team:platform", "tier:prod"]
        });
        validate_observability_exporter(&v).unwrap();
    }

    #[test]
    fn exporter_datadog_accepts_every_allow_list_site() {
        for site in [
            "datadoghq.com",
            "us3.datadoghq.com",
            "us5.datadoghq.com",
            "datadoghq.eu",
            "ap1.datadoghq.com",
            "ap2.datadoghq.com",
            "ddog-gov.com",
        ] {
            let v = json!({
                "name": "x",
                "kind": "datadog",
                "site": site,
                "credential_ref": "r",
                "service": "s"
            });
            validate_observability_exporter(&v)
                .unwrap_or_else(|e| panic!("site {site:?} must validate: {e:?}"));
        }
    }

    #[test]
    fn exporter_datadog_rejects_non_allow_list_site() {
        // A plausible-looking but unsupported / spoofed site must be rejected —
        // no exfil to an arbitrary `http-intake.logs.<host>`.
        for bad in [
            "evil.datadoghq.com.attacker.test",
            "datadoghq.org",
            "us9.datadoghq.com",
            "datadog.com",
            "datadoghq.com:443", // a port is NOT allowed on a real site
            "",
        ] {
            let v = json!({
                "name": "x",
                "kind": "datadog",
                "site": bad,
                "credential_ref": "r",
                "service": "s"
            });
            assert!(
                validate_observability_exporter(&v).is_err(),
                "site {bad:?} must be rejected by the allow-list"
            );
        }
    }

    #[test]
    fn exporter_datadog_allows_loopback_mock_site() {
        // The e2e points the DP at a local mock Datadog intake — bare host OR
        // host:port. The harness binds a FREE port, so `:port` must validate
        // (the prior exact-enum rejected it while the sink accepted it — #548).
        for site in ["mock-datadog", "127.0.0.1:54321", "localhost:8080"] {
            let v = json!({
                "name": "datadog-e2e",
                "kind": "datadog",
                "site": site,
                "credential_ref": "mock",
                "service": "ai-gateway"
            });
            validate_observability_exporter(&v)
                .unwrap_or_else(|e| panic!("loopback site {site:?} must validate: {e:?}"));
        }
    }

    #[test]
    fn exporter_datadog_requires_site_credential_service() {
        for missing in ["site", "credential_ref", "service"] {
            let mut v = json!({
                "name": "x",
                "kind": "datadog",
                "site": "datadoghq.com",
                "credential_ref": "r",
                "service": "s"
            });
            v.as_object_mut().unwrap().remove(missing);
            assert!(
                validate_observability_exporter(&v).is_err(),
                "missing `{missing}` must be rejected"
            );
        }
    }

    #[test]
    fn exporter_datadog_rejects_plaintext_api_key() {
        // No API-key field is allowed at the schema layer either —
        // `additionalProperties: false` rejects it before serde runs.
        let v = json!({
            "name": "x",
            "kind": "datadog",
            "site": "datadoghq.com",
            "credential_ref": "r",
            "service": "s",
            "api_key": "DDSECRET"
        });
        assert!(validate_observability_exporter(&v).is_err());
    }

    #[test]
    fn exporter_datadog_content_capture_fields() {
        let base = |extra: serde_json::Value| {
            let mut v = json!({
                "name": "x",
                "kind": "datadog",
                "site": "datadoghq.com",
                "credential_ref": "r",
                "service": "s"
            });
            let obj = v.as_object_mut().unwrap();
            for (k, val) in extra.as_object().unwrap() {
                obj.insert(k.clone(), val.clone());
            }
            v
        };
        // Opt-in content capture validates.
        validate_observability_exporter(&base(
            json!({ "content_mode": "full", "content_max_bytes": 4096 }),
        ))
        .unwrap();
        // Unknown content_mode is rejected.
        assert!(
            validate_observability_exporter(&base(json!({ "content_mode": "verbose" }))).is_err()
        );
        // content_max_bytes must be a positive integer (min 1).
        assert!(validate_observability_exporter(&base(json!({ "content_max_bytes": 0 }))).is_err());
        // content_max_bytes is capped at 1 MiB (Datadog per-log limit).
        assert!(
            validate_observability_exporter(&base(json!({ "content_max_bytes": 1_048_577 })))
                .is_err()
        );
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
        // "day" graduated into the enum (#771); "week" stays out.
        let v = json!({
            "name": "bad",
            "scope": "team",
            "scope_ref": "x",
            "window": "week",
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

    // ---- rate_limit_policy conditional form (AISIX-Cloud#892) ----

    #[test]
    fn rate_limit_policy_conditional_form_passes_both_validator_sets() {
        let v = json!({
            "name": "algo-team-premium",
            "conditions": [
                { "dimension": "team", "operator": "in", "value": ["t-1"] },
                { "logic": "or", "children": [
                    { "dimension": "model_name", "operator": "~~", "value": "^gpt-4\\.1" },
                    { "dimension": "provider", "operator": "==", "value": "anthropic" }
                ]}
            ],
            "group_by": ["team"],
            "limits": { "rpm": 1000, "tpm": 1000000 },
            "action": "reject"
        });
        validate_rate_limit_policy(&v).unwrap();
        validate_rate_limit_policy_lenient(&v).unwrap();
    }

    #[test]
    fn rate_limit_policy_conditional_minimal_is_just_limits() {
        // conditions/group_by/action are all optional — `limits` alone
        // is a valid "cap every request in the env" policy.
        let v = json!({
            "name": "env-wide",
            "limits": { "concurrency": 10 }
        });
        validate_rate_limit_policy(&v).unwrap();
    }

    #[test]
    fn rate_limit_policy_rejects_mixed_forms() {
        // A row carrying both a classic field and a conditional field
        // fails the injected oneOf in BOTH validator sets — an old DP
        // must never half-enforce such a row.
        let v = json!({
            "name": "mixed",
            "scope": "team",
            "scope_ref": "x",
            "window": "minute",
            "max_requests": 10,
            "limits": { "rpm": 5 }
        });
        assert!(validate_rate_limit_policy(&v).is_err());
        assert!(validate_rate_limit_policy_lenient(&v).is_err());
    }

    #[test]
    fn rate_limit_policy_condition_node_unknown_field_writes_red_reads_reported() {
        // ConditionNode is #[serde(untagged)]: serde silently swallows
        // unknown fields inside untagged content, so the schema closure is
        // the only guard on write — and `unknown_field_paths` is the only
        // report on read, since serde_ignored cannot see in there either.
        let v = json!({
            "name": "sneaky",
            "conditions": [
                { "dimension": "team", "operator": "==", "value": "t-1", "extra": 1 }
            ],
            "limits": { "rpm": 5 }
        });
        assert!(validate_rate_limit_policy(&v).is_err());
        validate_rate_limit_policy_lenient(&v).expect("a stored policy keeps limiting");
        assert_eq!(
            unknown_field_paths("rate_limit_policy", &v),
            vec!["conditions.0.extra".to_string()]
        );
    }

    #[test]
    fn rate_limit_policy_rejects_unknown_dimension_and_operator() {
        let bad_dim = json!({
            "name": "bad",
            "conditions": [ { "dimension": "region", "operator": "==", "value": "us" } ],
            "limits": { "rpm": 5 }
        });
        assert!(validate_rate_limit_policy(&bad_dim).is_err());
        let bad_op = json!({
            "name": "bad",
            "conditions": [ { "dimension": "team", "operator": "matches", "value": "t" } ],
            "limits": { "rpm": 5 }
        });
        assert!(validate_rate_limit_policy(&bad_op).is_err());
    }

    #[test]
    fn rate_limit_policy_classic_rows_unchanged_by_892() {
        // The exact pre-#892 shape keeps validating — stored rows are
        // never rewritten, so the classic branch must stay byte-stable.
        let v = json!({
            "name": "team-acme-tpm",
            "scope": "team",
            "scope_ref": "11111111-1111-1111-1111-111111111111",
            "window": "minute",
            "max_requests": 1000,
            "max_tokens": 1000000
        });
        validate_rate_limit_policy(&v).unwrap();
        validate_rate_limit_policy_lenient(&v).unwrap();
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

    // ---- renamed-field dual acceptance ----
    //
    // provider_key `secret`→`api_key` and mcp_server / a2a_agent
    // `display_name`→`name`: the generated schema must accept both
    // spellings (stored documents and current control-plane writes still
    // carry the former names), require at least one, and keep rejecting
    // unknown fields. A document carrying both spellings passes the
    // schema and is rejected by serde's duplicate-field check — that
    // split is pinned by the loader tests in `aisix-etcd`.

    #[test]
    fn provider_key_accepts_both_credential_spellings() {
        validate_provider_key(&json!({"display_name": "x", "api_key": "sk-x"})).unwrap();
        validate_provider_key(&json!({"display_name": "x", "secret": "sk-x"})).unwrap();
    }

    #[test]
    fn provider_key_requires_at_least_one_credential_spelling() {
        assert!(validate_provider_key(&json!({"display_name": "x"})).is_err());
    }

    #[test]
    fn provider_key_former_spelling_keeps_field_constraints() {
        // The former property clones the canonical one, so `minLength: 1`
        // keeps applying under either name.
        assert!(validate_provider_key(&json!({"display_name": "x", "api_key": ""})).is_err());
        assert!(validate_provider_key(&json!({"display_name": "x", "secret": ""})).is_err());
    }

    #[test]
    fn provider_key_schema_passes_document_with_both_spellings() {
        // Schema-layer half of the both-spellings corner: `anyOf` admits
        // the document; the serde layer rejects it as a duplicate field.
        validate_provider_key(&json!({"display_name": "x", "api_key": "a", "secret": "b"}))
            .unwrap();
    }

    #[test]
    fn mcp_server_accepts_both_label_spellings() {
        validate_mcp_server(&json!({"name": "github", "url": "https://x/mcp"})).unwrap();
        validate_mcp_server(&json!({"display_name": "github", "url": "https://x/mcp"})).unwrap();
        assert!(validate_mcp_server(&json!({"url": "https://x/mcp"})).is_err());
    }

    #[test]
    fn a2a_agent_accepts_both_label_spellings() {
        validate_a2a_agent(&json!({"name": "invoice", "url": "https://x/a2a"})).unwrap();
        validate_a2a_agent(&json!({"display_name": "invoice", "url": "https://x/a2a"})).unwrap();
        assert!(validate_a2a_agent(&json!({"url": "https://x/a2a"})).is_err());
    }

    #[test]
    fn schema_error_for_missing_name_does_not_echo_the_document() {
        // The renamed-field `anyOf` sits at the document root, and an
        // unmasked anyOf failure message interpolates the entire failing
        // instance — which for these resources can carry a live upstream
        // credential. `validate` masks instance values, so a name-less
        // document's error must not echo its `secret`.
        let err = validate_mcp_server(&json!({
            "url": "https://x/mcp",
            "auth_type": "bearer",
            "secret": "tok-sensitive"
        }))
        .expect_err("name-less document must fail");
        assert!(
            !err.to_string().contains("tok-sensitive"),
            "validation error must not echo credential values; got: {err}"
        );
    }

    #[test]
    fn renamed_field_acceptance_keeps_unknown_fields_rejected() {
        // The dual-name transform must not loosen `additionalProperties`.
        assert!(
            validate_provider_key(&json!({"display_name": "x", "api_key": "k", "rogue": 1}))
                .is_err()
        );
        assert!(validate_mcp_server(
            &json!({"name": "github", "url": "https://x/mcp", "rogue": 1})
        )
        .is_err());
        assert!(validate_a2a_agent(
            &json!({"name": "invoice", "url": "https://x/a2a", "rogue": 1})
        )
        .is_err());
    }

    // ---- mcp_server schema tests (#666 timeout_ms guard) ----

    #[test]
    fn mcp_server_minimal_passes() {
        // `timeout_ms` is optional; omitting it must validate.
        let v = json!({
            "display_name": "github",
            "url": "https://api.example.com/mcp"
        });
        validate_mcp_server(&v).unwrap();
    }

    #[test]
    fn mcp_server_accepts_positive_timeout_ms() {
        let v = json!({
            "display_name": "github",
            "url": "https://api.example.com/mcp",
            "timeout_ms": 1
        });
        validate_mcp_server(&v).unwrap();
    }

    #[test]
    fn mcp_server_accepts_api_key_auth() {
        let v = json!({
            "display_name": "github",
            "url": "https://api.example.com/mcp",
            "auth_type": "api_key",
            "secret": "k-123"
        });
        validate_mcp_server(&v).unwrap();
    }

    #[test]
    fn mcp_server_accepts_oauth2_auth_with_client_fields() {
        let v = json!({
            "display_name": "github",
            "url": "https://api.example.com/mcp",
            "auth_type": "oauth2",
            "secret": "client-secret",
            "client_id": "cid",
            "token_url": "https://auth.example.com/oauth/token",
            "scopes": ["read", "write"]
        });
        validate_mcp_server(&v).unwrap();
    }

    #[test]
    fn mcp_server_rejects_unknown_auth_type_and_bad_scopes_shape() {
        // The `auth_type` set is closed: near-misses like `oauth` must fail.
        let v = json!({
            "display_name": "x",
            "url": "https://x/mcp",
            "auth_type": "oauth"
        });
        assert!(validate_mcp_server(&v).is_err());

        // `scopes` is an array of strings, not a single space-joined string.
        let v = json!({
            "display_name": "x",
            "url": "https://x/mcp",
            "auth_type": "oauth2",
            "secret": "s",
            "client_id": "cid",
            "token_url": "https://auth/token",
            "scopes": "read write"
        });
        assert!(validate_mcp_server(&v).is_err());
    }

    #[test]
    fn mcp_server_schema_enforces_credential_coupling() {
        // This assertion is the inverse of what it used to be, deliberately.
        // The coupling (oauth2 ⇒ client_id + secret + token_url) used to be
        // left to write paths, on the reasoning that an incomplete row should
        // still load and degrade at runtime. That reasoning depended on a write
        // path existing to catch it; with resource writes removed from this
        // gateway, leaving the schema permissive means nothing checks the
        // coupling at all on the declarative and etcd paths.
        //
        // Rejecting at load is also the more diagnosable of the two failures: a
        // rejected row is named in `GET /status/config`'s `rejected` array,
        // whereas a loaded-but-degraded server silently serves no tools.
        let v = json!({
            "display_name": "x",
            "url": "https://x/mcp",
            "auth_type": "oauth2"
        });
        assert!(validate_mcp_server(&v).is_err());

        // The complete set still validates.
        let v = json!({
            "display_name": "x",
            "url": "https://x/mcp",
            "auth_type": "oauth2",
            "secret": "cs",
            "client_id": "cid",
            "token_url": "https://auth/token"
        });
        validate_mcp_server(&v).unwrap();
    }

    // ---- strict-write / lenient-read split (issue #871) ----

    #[test]
    fn write_rejection_names_the_unknown_field_behind_the_one_of() {
        // A typo in a resources file must say which field it was. The
        // guardrail and exporter documents are `oneOf`s, so `jsonschema`
        // reports only "not valid under any of the schemas" and the author
        // is left guessing.
        let err = validate_guardrail(&json!({
            "name": "p", "kind": "pii",
            "custom_patterns": [{"name": "n", "regex": "x", "replacment": "***"}]
        }))
        .expect_err("the write contract rejects the typo");
        assert!(
            err.message.contains("custom_patterns.0.replacment"),
            "unhelpful message: {}",
            err.message
        );

        let err = validate_observability_exporter(&json!({
            "name": "o", "kind": "otlp_http",
            "endpoint": "https://otel.example/v1/traces", "timout_ms": 5
        }))
        .expect_err("the write contract rejects the typo");
        assert!(
            err.message.contains("timout_ms"),
            "unhelpful message: {}",
            err.message
        );

        // A document that ALSO violates something else keeps the original
        // error: replacing it would leave `path` and `message` describing
        // different fields.
        let err = validate_guardrail(&json!({
            "name": "", "kind": "pii",
            "custom_patterns": [{"name": "n", "regex": "x", "replacment": "***"}]
        }))
        .expect_err("an empty name is still a violation");
        assert!(
            !err.message.contains("replacment"),
            "the other problem must win: {}",
            err.message
        );
    }

    #[test]
    fn lenient_set_carries_no_closure_at_any_depth() {
        // The mechanical guarantee behind "an additive optional field is
        // ignored, not fatal": there is nowhere left in the read set for a
        // closure to hide. A single `additionalProperties: false` surviving
        // in here — placed by a producer, or emitted by `schemars` from a
        // nested `#[serde(deny_unknown_fields)]` struct — is one whole
        // stored row lost the first time a newer control plane writes a
        // field under it.
        for resource in RESOURCES {
            let mut found = Vec::new();
            closure_paths(&resource_root_schema(resource, false), "", &mut found);
            assert!(
                found.is_empty(),
                "{resource} read schema still closes {found:?}"
            );
        }
    }

    fn closure_paths(node: &Value, path: &str, out: &mut Vec<String>) {
        match node {
            Value::Object(obj) => {
                if obj.get("additionalProperties") == Some(&Value::Bool(false)) {
                    out.push(path.to_string());
                }
                for (key, child) in obj {
                    closure_paths(child, &format!("{path}/{key}"), out);
                }
            }
            Value::Array(items) => {
                for (index, item) in items.iter().enumerate() {
                    closure_paths(item, &format!("{path}/{index}"), out);
                }
            }
            _ => {}
        }
    }

    /// Every nested object the read path used to close, one document each,
    /// carrying a field this build does not know. Before this pair of passes
    /// existed, every one of these was a skipped row on a data plane one
    /// release behind its control plane — a guardrail that stopped
    /// enforcing, an exporter that stopped exporting.
    #[test]
    fn formerly_closed_nested_objects_load_on_read_and_are_reported() {
        let bedrock = |credentials: Value, latency: Value| {
            json!({
                "name": "b", "kind": "bedrock",
                "guardrail_id": "gr-1", "guardrail_version": "1", "region": "us-east-1",
                "aws_credentials": credentials,
                "latency_mode": latency,
            })
        };
        let serial = json!({"kind": "serial"});
        let credentials = json!({"kind": "static", "access_key_id": "a", "secret_access_key": "b"});
        let cases: Vec<(&str, Value, &str)> = vec![
            (
                "guardrail",
                json!({
                    "name": "p", "kind": "pii",
                    "custom_patterns": [{"name": "n", "regex": "x", "future_knob": 1}]
                }),
                "custom_patterns.0.future_knob",
            ),
            (
                "guardrail",
                json!({
                    "name": "p", "kind": "pii",
                    "detectors": [{"type": "email", "future_knob": 1}]
                }),
                "detectors.0.future_knob",
            ),
            (
                "guardrail",
                json!({
                    "name": "p", "kind": "presidio",
                    "analyzer_url": "http://a", "anonymizer_url": "http://b",
                    "entities": [{"type": "EMAIL_ADDRESS", "future_knob": 1}]
                }),
                "entities.0.future_knob",
            ),
            (
                "guardrail",
                json!({
                    "name": "k", "kind": "keyword",
                    "patterns": [{"kind": "literal", "value": "x", "future_knob": 1}]
                }),
                "patterns.0.future_knob",
            ),
            (
                "guardrail",
                bedrock(
                    json!({
                        "kind": "static", "access_key_id": "a", "secret_access_key": "b",
                        "future_knob": 1
                    }),
                    serial.clone(),
                ),
                "aws_credentials.future_knob",
            ),
            (
                "guardrail",
                bedrock(
                    credentials.clone(),
                    json!({"kind": "serial", "future_knob": 1}),
                ),
                "latency_mode.future_knob",
            ),
            (
                "guardrail",
                bedrock(
                    credentials,
                    json!({"kind": "timed", "timeout_ms": 200, "future_knob": 1}),
                ),
                "latency_mode.future_knob",
            ),
            (
                // Not a nested object at all: the guardrail ROOT, where the
                // guard was `deny_unknown_fields` on the flattened kind
                // config rather than a schema closure. Same blast radius.
                "guardrail",
                json!({"name": "k", "kind": "keyword", "patterns": [], "future_knob": 1}),
                "future_knob",
            ),
            (
                "observability_exporter",
                json!({
                    "name": "o", "kind": "otlp_http",
                    "endpoint": "https://otel.example/v1/traces", "future_knob": 1
                }),
                "future_knob",
            ),
            (
                "observability_exporter",
                json!({
                    "name": "o", "kind": "aliyun_sls",
                    "endpoint": "ap-southeast-3.log.aliyuncs.com",
                    "project": "p", "logstore": "l", "credential_ref": "r",
                    "future_knob": 1
                }),
                "future_knob",
            ),
            (
                "observability_exporter",
                json!({
                    "name": "o", "kind": "object_store", "provider": "s3",
                    "bucket": "b", "prefix": "p", "credential_ref": "r",
                    "future_knob": 1
                }),
                "future_knob",
            ),
            (
                "observability_exporter",
                json!({
                    "name": "o", "kind": "datadog", "site": "datadoghq.com",
                    "credential_ref": "r", "service": "s", "future_knob": 1
                }),
                "future_knob",
            ),
            (
                "rate_limit_policy",
                json!({
                    "name": "r",
                    "conditions": [
                        {"logic": "and", "future_knob": 1, "children": [
                            {"dimension": "team", "operator": "==", "value": "t"}
                        ]}
                    ],
                    "limits": {"rpm": 5}
                }),
                "conditions.0.future_knob",
            ),
            (
                "model",
                json!({
                    "display_name": "s",
                    "semantic": {
                        "embedding_model": "e",
                        "routes": [{"name": "a", "target": "m", "examples": ["x"]}],
                        "default": "d",
                        "match": {"threshold": 0.5},
                        "on_embedding_failure": {"target": "t", "future_knob": 1}
                    }
                }),
                "semantic.on_embedding_failure.future_knob",
            ),
        ];

        for (resource, document, field) in cases {
            let strict = strict_validator(resource);
            let lenient = lenient_validator(resource);
            assert!(
                validate(strict, &document).is_err(),
                "{resource}/{field}: the write contract must keep rejecting the typo"
            );
            validate(lenient, &document).unwrap_or_else(|err| {
                panic!("{resource}/{field}: stored row must keep loading, got {err}")
            });
            assert_eq!(
                unknown_field_paths(resource, &document),
                vec![field.to_string()],
                "{resource}/{field}: tolerated is not the same as silent"
            );
        }
    }

    #[test]
    fn opening_the_read_set_does_not_disturb_one_of_selection() {
        // The reason opening was safe to begin with: every union among the
        // formerly-closed objects discriminates on a `kind` const, not on
        // the closure. The sharpest case is a document whose extra field is
        // exactly a SIBLING branch's field — under `oneOf`, a second
        // matching branch would fail the document just as hard as the
        // closure used to.
        let sibling_field = json!({
            "name": "b", "kind": "bedrock",
            "guardrail_id": "gr-1", "guardrail_version": "1", "region": "us-east-1",
            "aws_credentials": {"kind": "static", "access_key_id": "a", "secret_access_key": "b"},
            // `timeout_ms` belongs to the `timed` branch; `serial` no longer
            // rejects it, and must still be the only branch that matches.
            "latency_mode": {"kind": "serial", "timeout_ms": 200}
        });
        validate_guardrail_lenient(&sibling_field).expect("exactly one branch still matches");
        assert!(validate_guardrail(&sibling_field).is_err());
        let parsed: crate::models::Guardrail =
            serde_json::from_value(sibling_field.clone()).expect("row deserializes");
        match &parsed.config {
            crate::models::GuardrailKind::Bedrock(cfg) => assert!(
                matches!(cfg.latency_mode, crate::models::BedrockLatencyMode::Serial),
                "the `kind` const, not the closure, picks the branch"
            ),
            other => panic!("wrong kind: {}", other.kind_str()),
        }
        // A sibling branch declares `timeout_ms`, so the report stays quiet:
        // it is a malformed document, not a field from a newer build, and
        // the strict path above is what says so.
        assert!(unknown_field_paths("guardrail", &sibling_field).is_empty());

        // Same at the top level: a `keyword` guardrail carrying a `pii`
        // field still resolves to `keyword`, because every other branch
        // pins its own `kind`. Cross-kind leakage is a malformed document,
        // not a newer build's field, so the write path rejects it and the
        // report stays quiet about it (see `unknown_field_paths`).
        let cross_kind = json!({
            "name": "k", "kind": "keyword", "patterns": [],
            "detectors": [{"type": "email"}]
        });
        validate_guardrail_lenient(&cross_kind).expect("exactly one branch still matches");
        assert!(validate_guardrail(&cross_kind).is_err());
        let parsed: crate::models::Guardrail =
            serde_json::from_value(cross_kind.clone()).expect("row deserializes");
        assert_eq!(parsed.config.kind_str(), "keyword");
        assert!(unknown_field_paths("guardrail", &cross_kind).is_empty());
    }

    #[test]
    fn valid_documents_report_no_unknown_fields() {
        // The other half of the report's contract: a document this build
        // fully understands must stay GREEN. A field the schema cannot see
        // but serde consumes — a `#[schemars(skip)]` tombstone — would show
        // up here as a permanent YELLOW on every row, which is why the
        // resources `unknown_field_paths` covers must not carry one.
        let documents: Vec<(&str, Value)> = vec![
            (
                "guardrail",
                json!({
                    "name": "p", "kind": "pii", "enabled": true, "hook_point": "both",
                    "fail_open": false, "enforcement_mode": "monitor", "mandatory": true,
                    "direction": "input", "created_at": "2026-01-01T00:00:00Z",
                    "detectors": [{"type": "email", "action": "mask"}],
                    "custom_patterns": [
                        {"name": "n", "regex": "x(y)", "action": "mask", "replacement": "***"}
                    ],
                    "default_action": "mask"
                }),
            ),
            (
                "observability_exporter",
                json!({
                    "name": "o", "enabled": true, "kind": "otlp_http",
                    "endpoint": "https://otel.example/v1/traces",
                    "headers": {"x-team": "abc"}
                }),
            ),
            (
                "rate_limit_policy",
                json!({
                    "name": "r",
                    "conditions": [
                        {"logic": "and", "children": [
                            {"dimension": "team", "operator": "==", "value": "t"}
                        ]}
                    ],
                    "limits": {"rpm": 5}
                }),
            ),
            (
                "model",
                json!({
                    "display_name": "d", "provider": "openai", "model_name": "gpt-4o",
                    "provider_key_id": "11111111-1111-1111-1111-111111111111"
                }),
            ),
        ];
        for (resource, document) in documents {
            validate(strict_validator(resource), &document)
                .unwrap_or_else(|err| panic!("{resource} fixture must be valid: {err}"));
            assert!(
                unknown_field_paths(resource, &document).is_empty(),
                "{resource}: a fully known document must not report unknown fields"
            );
        }
    }

    fn strict_validator(resource: &str) -> &'static Validator {
        match resource {
            "guardrail" => &SCHEMAS.guardrail,
            "observability_exporter" => &SCHEMAS.observability_exporter,
            "rate_limit_policy" => &SCHEMAS.rate_limit_policy,
            "model" => &SCHEMAS.model,
            other => panic!("no strict validator wired for {other}"),
        }
    }

    fn lenient_validator(resource: &str) -> &'static Validator {
        match resource {
            "guardrail" => &LENIENT_SCHEMAS.guardrail,
            "observability_exporter" => &LENIENT_SCHEMAS.observability_exporter,
            "rate_limit_policy" => &LENIENT_SCHEMAS.rate_limit_policy,
            "model" => &LENIENT_SCHEMAS.model,
            other => panic!("no lenient validator wired for {other}"),
        }
    }

    #[test]
    fn lenient_set_tolerates_unknown_fields_strict_set_rejects() {
        let v = json!({
            "key_hash": "9df37f5e7cbc3c391d872742b5f286c242e733a09add9eeaa4d26a599bd90b20",
            "allowed_models": ["a"],
            "future_field": true
        });
        assert!(validate_apikey(&v).is_err(), "write contract stays strict");
        validate_apikey_lenient(&v).expect("read contract tolerates unknown fields");
    }

    #[test]
    fn lenient_set_still_enforces_every_other_constraint() {
        // Missing required field.
        assert!(validate_apikey_lenient(&json!({"allowed_models": []})).is_err());
        // Unknown enum value.
        let v = json!({
            "display_name": "r",
            "routing": {"strategy": "quantum", "targets": [{"model": "a"}]}
        });
        assert!(validate_model_lenient(&v).is_err());
        // Range violation.
        let v = json!({
            "display_name": "", "provider": "openai",
            "model_name": "g", "provider_key_id": "pk"
        });
        assert!(validate_model_lenient(&v).is_err());
    }

    #[test]
    fn lenient_set_opens_the_producers_closures_and_reports_them_instead() {
        // These closures used to hold on the read path too, on the argument
        // that serde cannot report ignored fields inside tagged-enum
        // content and a silent tolerance is worse than a rejection. But a
        // rejection here is the whole row: an exporter that stops
        // exporting, a guardrail that stops enforcing, because a newer
        // control plane added one optional field. The read path opens them
        // and `unknown_field_paths` supplies the report serde cannot.
        let exporter = json!({
            "name": "o", "kind": "otlp_http",
            "endpoint": "https://otel.example/v1/traces",
            "smuggled_secret": "sk-x"
        });
        assert!(validate_observability_exporter(&exporter).is_err());
        validate_observability_exporter_lenient(&exporter)
            .expect("a stored exporter keeps exporting");
        assert_eq!(
            unknown_field_paths("observability_exporter", &exporter),
            vec!["smuggled_secret".to_string()]
        );

        // Same for the guardrail tagged sub-enums.
        let guardrail = json!({
            "name": "kw", "kind": "keyword",
            "patterns": [{"kind": "literal", "value": "x", "extra": 1}]
        });
        assert!(validate_guardrail(&guardrail).is_err());
        validate_guardrail_lenient(&guardrail).expect("a stored guardrail keeps enforcing");
        assert_eq!(
            unknown_field_paths("guardrail", &guardrail),
            vec!["patterns.0.extra".to_string()]
        );
    }

    #[test]
    fn on_embedding_failure_object_variant_writes_red_reads_reported() {
        // `OnEmbeddingFailure` is untagged with an object variant: serde
        // buffers untagged content and silently swallows unknown fields
        // inside it — invisible to serde_ignored too. The producer closes
        // the object branch so the typo is caught on write; on read the
        // row loads and `unknown_field_paths` names the field.
        let v = json!({
            "display_name": "prod-chat",
            "semantic": {
                "embedding_model": "e",
                "routes": [{"name": "a", "target": "m", "examples": ["x"]}],
                "default": "d",
                "match": {"threshold": 0.5},
                "on_embedding_failure": {"target": "t", "sneaky": 1}
            }
        });
        assert!(validate_model(&v).is_err());
        validate_model_lenient(&v).expect("a stored router keeps routing");
        assert_eq!(
            unknown_field_paths("model", &v),
            vec!["semantic.on_embedding_failure.sneaky".to_string()]
        );

        // The legitimate shapes keep validating on both paths.
        let ok = json!({
            "display_name": "prod-chat",
            "semantic": {
                "embedding_model": "e",
                "routes": [{"name": "a", "target": "m", "examples": ["x"]}],
                "default": "d",
                "match": {"threshold": 0.5},
                "on_embedding_failure": {"target": "t"}
            }
        });
        validate_model(&ok).unwrap();
        validate_model_lenient(&ok).unwrap();
    }

    #[test]
    fn mcp_server_rejects_zero_timeout_ms() {
        // A zero deadline times out every upstream op instantly and silently
        // drops the server from `tools/list`; reject it at the schema layer
        // (enforced on both the Admin write-path and the etcd loader).
        let v = json!({
            "display_name": "github",
            "url": "https://api.example.com/mcp",
            "timeout_ms": 0
        });
        assert!(validate_mcp_server(&v).is_err());
    }

    #[test]
    fn model_dead_knob_error_names_the_field_and_kind() {
        // The five-branch `oneOf` makes every branch fail, so the raw
        // jsonschema error is the root-level "not valid under any of the
        // schemas". The dead knob is the case the strict path exists to
        // catch, so it must be named.
        let group = json!({
            "display_name": "g",
            "routing": {"strategy": "failover", "targets": [{"model": "m"}]},
            "retries": 3,
            "cost": {"input_per_1k": 0.5, "output_per_1k": 1.5}
        });
        let msg = validate_model(&group).unwrap_err().message;
        assert!(msg.contains("`cost`"), "{msg}");
        assert!(msg.contains("`retries`"), "{msg}");
        assert!(msg.contains("model group"), "{msg}");
        assert!(
            !msg.contains("oneOf"),
            "generic message should be replaced: {msg}"
        );

        let ensemble = json!({
            "display_name": "e",
            "ensemble": {"panel": [{"model": "m"}], "judge": {"model": "m"}},
            "timeout": 1000
        });
        let msg = validate_model(&ensemble).unwrap_err().message;
        assert!(msg.contains("`timeout`"), "{msg}");
        assert!(msg.contains("ensemble"), "{msg}");

        let semantic = json!({
            "display_name": "s",
            "semantic": {
                "embedding_model": "e",
                "routes": [{"name": "a", "target": "m", "examples": ["x"]}],
                "default": "d",
                "match": {"threshold": 0.5},
                "on_embedding_failure": {"target": "t"}
            },
            "auto_prompt_caching": {"enabled": true}
        });
        let msg = validate_model(&semantic).unwrap_err().message;
        assert!(msg.contains("`auto_prompt_caching`"), "{msg}");
        assert!(msg.contains("semantic router"), "{msg}");
    }

    #[test]
    fn model_non_dead_knob_failures_keep_the_generic_message() {
        // A failure that is NOT a dead knob must not be relabelled: the
        // enrichment is best-effort and only speaks for the case it can
        // prove. An unknown field on a direct model is rejected by
        // `additionalProperties: false`, and strip_kind_inapplicable has
        // nothing to say about it.
        let unknown = json!({
            "display_name": "d",
            "provider": "openai",
            "model_name": "gpt-4o-mini",
            "provider_key_id": "pk",
            "definitely_not_a_field": 1
        });
        let msg = validate_model(&unknown).unwrap_err().message;
        assert!(!msg.contains("not accepted on a"), "{msg}");

        // A dead knob on a DIRECT model is not dead at all — it resolves
        // there — so a direct model carrying `retries` must still VALIDATE.
        let direct = json!({
            "display_name": "d",
            "provider": "openai",
            "model_name": "gpt-4o-mini",
            "provider_key_id": "pk",
            "retries": 3
        });
        validate_model(&direct).unwrap();
    }

    #[test]
    fn model_dead_knob_error_carries_no_instance_values() {
        // The masking contract: validation errors reach logs, the
        // rejection buffer and admin 400 bodies, and model documents can
        // carry credentials. Only field NAMES may be added.
        let group = json!({
            "display_name": "g",
            "routing": {"strategy": "failover", "targets": [{"model": "m"}]},
            "cost": {"input_per_1k": 12345.678, "output_per_1k": 99999.111}
        });
        let msg = validate_model(&group).unwrap_err().message;
        assert!(msg.contains("`cost`"), "{msg}");
        assert!(!msg.contains("12345"), "instance value leaked: {msg}");
        assert!(!msg.contains("99999"), "instance value leaked: {msg}");
    }

    #[test]
    fn model_dead_knob_with_an_independent_failure_keeps_the_original_error() {
        // A dead knob is only named when it is the WHOLE story. Here the
        // group also has an empty display_name (minLength 1) — which
        // `Model` deserialises fine, so the enrichment is reachable.
        // Replacing the message would report `retries` while `path` still
        // points at /display_name: two different fields in one error.
        let v = json!({
            "display_name": "",
            "routing": {"strategy": "failover", "targets": [{"model": "m"}]},
            "retries": 3
        });
        let err = validate_model(&v).unwrap_err();
        assert_eq!(err.path, "/display_name", "{err:?}");
        assert!(
            !err.message.contains("`retries`"),
            "the independent failure must win: {err:?}"
        );
        assert!(err.message.contains("shorter than 1 character"), "{err:?}");

        // With the independent violation fixed, the dead knob is the whole
        // story again and gets named.
        let v = json!({
            "display_name": "g",
            "routing": {"strategy": "failover", "targets": [{"model": "m"}]},
            "retries": 3
        });
        let err = validate_model(&v).unwrap_err();
        assert!(err.message.contains("`retries`"), "{err:?}");
    }
}
