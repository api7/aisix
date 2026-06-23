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
use aisix_core::models::{EnsembleConfig, RateLimit, Routing};

fn main() {
    let out_dir = workspace_root().join("schemas").join("resources");
    fs::create_dir_all(&out_dir).expect("create schemas/resources dir");

    // Every resource with a runtime validator goes through the SAME
    // `*_root_schema()` producer the validator uses, so the published schema ==
    // the enforced schema by construction. `ensemble`/`rate_limit`/`routing`
    // have no standalone validator (they are nested struct types) so they dump
    // straight from the struct via `schema_for!`.
    dump_value(&out_dir, "api_key", schema::apikey_root_schema());
    dump_value(&out_dir, "cache_policy", schema::cache_policy_root_schema());
    dump_value(&out_dir, "model", schema::model_root_schema());
    dump_value(
        &out_dir,
        "rate_limit_policy",
        schema::rate_limit_policy_root_schema(),
    );
    dump_value(&out_dir, "provider_key", schema::provider_key_root_schema());
    dump_value(
        &out_dir,
        "observability_exporter",
        schema::observability_exporter_root_schema(),
    );
    dump_value(&out_dir, "guardrail", schema::guardrail_root_schema());
    dump_value(
        &out_dir,
        "guardrail_attachment",
        schema::guardrail_attachment_root_schema(),
    );

    dump::<EnsembleConfig>(&out_dir, "ensemble");
    dump::<RateLimit>(&out_dir, "rate_limit");
    dump::<Routing>(&out_dir, "routing");
}

fn dump<T: JsonSchema>(out_dir: &Path, name: &str) {
    // Serialize the `RootSchema` directly to preserve schemars' native key
    // ordering. (Routing through `serde_json::Value` would re-sort keys.)
    let mut json =
        serde_json::to_string_pretty(&schemars::schema_for!(T)).expect("serialize schema");
    json.push('\n');
    let path = out_dir.join(format!("{name}.schema.json"));
    fs::write(&path, json).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
    println!("wrote {}", path.display());
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
