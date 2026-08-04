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
    let out_dir = workspace_root().join("schemas").join("resources");
    fs::create_dir_all(&out_dir).expect("create schemas/resources dir");

    // Every resource with a runtime validator goes through the SAME
    // `resource_root_schema(name, strict: true)` producer the write-path
    // validators compile, so the published schema == the enforced write
    // contract by construction. The published files deliberately carry the
    // STRICT shape: they document the Admin API write contract (unknown
    // fields are a 400) and the etcd loader's lenient read tolerance is a
    // runtime behavior, not a contract callers may write against.
    // `ensemble`/`rate_limit`/`routing` have no standalone validator (they
    // are nested struct types) so they dump straight from the struct via
    // `schema_for!`, closed the same way.
    for resource in [
        "api_key",
        "cache_policy",
        "model",
        "rate_limit_policy",
        "provider_key",
        "observability_exporter",
        "guardrail",
        "guardrail_attachment",
        "mcp_server",
        "mcp_policy",
        "a2a_agent",
        "oidc_provider",
    ] {
        dump_value(
            &out_dir,
            resource,
            schema::resource_root_schema(resource, true),
        );
    }

    dump::<EnsembleConfig>(&out_dir, "ensemble");
    dump::<RateLimit>(&out_dir, "rate_limit");
    dump::<Routing>(&out_dir, "routing");
    dump::<Semantic>(&out_dir, "semantic");
    dump::<EmbeddingConfig>(&out_dir, "embedding");
}

fn dump<T: JsonSchema>(out_dir: &Path, name: &str) {
    // Serialize the `RootSchema` directly to preserve schemars' native key
    // ordering. (Routing through `serde_json::Value` would re-sort keys.)
    // These nested types belong to closed resources, so re-close the root
    // and every struct-shaped definition on the typed schema — the same
    // strictness `schema::close_unknown_fields` applies to the resource
    // documents, kept typed here so the key order stays schemars-native.
    let mut root = schemars::schema_for!(T);
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
/// (one that lists `properties`), unless it already pins a value.
fn close_object_schema(schema: &mut schemars::schema::SchemaObject) {
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
