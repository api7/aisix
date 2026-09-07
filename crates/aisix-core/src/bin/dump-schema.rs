//! Emit canonical JSON Schema files for `aisix-core` resource types.
//!
//! Invocation:
//!
//! ```bash
//! cargo run -p aisix-core --bin dump-schema
//! ```
//!
//! Writes one file per top-level resource into
//! `<workspace-root>/schemas/resources/<name>.schema.json`. Each file
//! is a self-contained JSON Schema draft-07 document (the default of
//! `schemars` 0.8) — nested types live in the `definitions/` section
//! of the same document, no cross-file `$ref` required.
//!
//! Every run writes BOTH published sets, under the same file names:
//!
//! - `schemas/resources/` — the **strict** write contract
//!   (`resource_root_schema(name, true)`), what `aisix validate` and the
//!   resources-file source enforce.
//! - `schemas/resources-lenient/` — the **read** contract
//!   (`resource_root_schema(name, false)`), the schema the etcd loader
//!   actually validates stored documents against. It is free of
//!   `additionalProperties: false` at every depth, so a document written by
//!   a newer control plane loads with its extra fields ignored instead of
//!   the whole row being skipped. Published so a consumer that needs to know
//!   what this build will LOAD can read it instead of deriving it from the
//!   strict files.
//!
//! Re-run after modifying any resource struct in
//! `crates/aisix-core/src/models/`. CI runs this binary and rejects PRs
//! that leave `schemas/` out of date (drift check, follow-up PR).
//!
//! Downstream consumers:
//!
//! - `crates/aisix-admin/src/openapi.rs` — refactor target: replace
//!   inline schema objects in the hand-written OpenAPI doc with
//!   `$ref` into these files (follow-up PR).
//! - `api7/AISIX-Cloud` — pulls these files (via submodule or pinned
//!   tag) to drive cp-api request validation and dashboard form
//!   generation. Refs api7/ai-gateway#304 (#1).

use std::fs;
use std::path::{Path, PathBuf};

use schemars::JsonSchema;

use aisix_core::models::schema;
use aisix_core::models::{EmbeddingConfig, EnsembleConfig, RateLimit, Routing, Semantic};

fn main() {
    let schemas_dir = workspace_root().join("schemas");
    let out_dir = schemas_dir.join("resources");
    let lenient_dir = schemas_dir.join("resources-lenient");
    fs::create_dir_all(&out_dir).expect("create schemas/resources dir");
    fs::create_dir_all(&lenient_dir).expect("create schemas/resources-lenient dir");

    // Every resource with a runtime validator goes through the SAME
    // `resource_root_schema(name, strict: true)` producer the strict
    // validators compile, so the published schema == the enforced write
    // contract by construction. The published files deliberately carry the
    // STRICT shape: they document the declarative write contract (unknown
    // fields are rejected by `aisix validate` and the file source wherever
    // a resource closes them) and the
    // etcd loader's lenient read tolerance is published beside them, as
    // `schemas/resources-lenient/`, from the same producer with
    // `strict: false` — the exact value `LENIENT_SCHEMAS` compiles, so the
    // published read contract cannot drift from the enforced one either.
    // `ensemble`/`rate_limit`/`routing` have no standalone validator (they
    // are nested struct types) so they dump straight from the struct via
    // `schema_for!`, closed the same way.
    for resource in schema::RESOURCES {
        dump_value(
            &out_dir,
            resource,
            schema::resource_root_schema(resource, true),
        );
        dump_value(
            &lenient_dir,
            resource,
            schema::resource_root_schema(resource, false),
        );
    }

    dump::<EnsembleConfig>(&out_dir, &lenient_dir, "ensemble");
    dump::<RateLimit>(&out_dir, &lenient_dir, "rate_limit");
    dump::<Routing>(&out_dir, &lenient_dir, "routing");
    dump::<Semantic>(&out_dir, &lenient_dir, "semantic");
    dump::<EmbeddingConfig>(&out_dir, &lenient_dir, "embedding");
}

fn dump<T: JsonSchema>(out_dir: &Path, lenient_dir: &Path, name: &str) {
    let mut root = schemars::schema_for!(T);

    // The lenient twin comes off the SAME producer, run through
    // `schema::open_unknown_fields` — the very pass `LENIENT_SCHEMAS`
    // compiles the resource roots with — before the closing pass below runs.
    // These nested types have no standalone loader validator (they are only
    // ever validated as part of the resource that embeds them), so the pass
    // is what ties the published file to the read contract.
    let mut lenient = serde_json::to_value(&root).expect("serialize schema");
    schema::open_unknown_fields(&mut lenient);
    dump_value(lenient_dir, name, lenient);

    // Serialize the `RootSchema` directly to preserve schemars' native key
    // ordering. (Routing through `serde_json::Value` would re-sort keys.)
    // These nested types belong to closed resources, so re-close the root
    // and every struct-shaped definition on the typed schema — the same
    // strictness `schema::close_unknown_fields` applies to the resource
    // documents, kept typed here so the key order stays schemars-native.
    close_object_schema(&mut root.schema);
    for def in root.definitions.values_mut() {
        if let schemars::schema::Schema::Object(obj) = def {
            close_object_schema(obj);
        }
    }
    let mut json = serde_json::to_string_pretty(&root).expect("serialize schema");
    json.push('\n');
    let path = out_dir.join(format!("{name}.schema.json"));
    fs::write(&path, json).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
    println!("wrote {}", path.display());
}

/// Insert `additionalProperties: false` on a struct-shaped schema object
/// (one that lists `properties`), unless it already pins a value. Recurses
/// into `anyOf` branches so an untagged enum's object variant closes too —
/// serde silently swallows unknown fields inside untagged content, so the
/// schema closure is the only non-silent guard there (the resource
/// producers apply the same rule, e.g. `OnEmbeddingFailure` in `model`).
fn close_object_schema(schema: &mut schemars::schema::SchemaObject) {
    if let Some(sub) = schema.subschemas.as_deref_mut() {
        if let Some(any_of) = sub.any_of.as_mut() {
            for branch in any_of.iter_mut() {
                if let schemars::schema::Schema::Object(b) = branch {
                    close_object_schema(b);
                }
            }
        }
    }
    let Some(object) = schema.object.as_deref_mut() else {
        return;
    };
    if !object.properties.is_empty() && object.additional_properties.is_none() {
        object.additional_properties = Some(Box::new(schemars::schema::Schema::Bool(false)));
    }
}

/// Write a pre-assembled schema `Value`. Used for resources whose canonical
/// schema is built by a dedicated producer rather than a bare `schema_for!`
/// (e.g. `model`, which injects the cross-field `oneOf`).
fn dump_value(out_dir: &Path, name: &str, schema: serde_json::Value) {
    let mut json = serde_json::to_string_pretty(&schema).expect("serialize schema");
    json.push('\n');
    let path = out_dir.join(format!("{name}.schema.json"));
    fs::write(&path, json).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
    println!("wrote {}", path.display());
}

/// Workspace root, derived from the `aisix-core` manifest directory.
///
/// `CARGO_MANIFEST_DIR` is `<root>/crates/aisix-core` — `parent()` twice
/// resolves to `<root>`. The path is baked in at compile time, so the
/// binary always targets the workspace it was built in (correct for an
/// in-tree code-generation tool; not meant to ship outside the repo).
fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("CARGO_MANIFEST_DIR has two ancestors")
        .to_path_buf()
}
