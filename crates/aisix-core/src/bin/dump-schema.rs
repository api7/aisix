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

use aisix_core::models::{
    ApiKey, CachePolicy, Guardrail, Model, ObservabilityExporter, ProviderKey, RateLimit,
    RateLimitPolicy, Routing,
};

fn main() {
    let out_dir = workspace_root().join("schemas").join("resources");
    fs::create_dir_all(&out_dir).expect("create schemas/resources dir");

    dump::<ApiKey>(&out_dir, "api_key");
    dump::<CachePolicy>(&out_dir, "cache_policy");
    dump::<Guardrail>(&out_dir, "guardrail");
    dump::<Model>(&out_dir, "model");
    dump::<ObservabilityExporter>(&out_dir, "observability_exporter");
    dump::<ProviderKey>(&out_dir, "provider_key");
    dump::<RateLimit>(&out_dir, "rate_limit");
    dump::<RateLimitPolicy>(&out_dir, "rate_limit_policy");
    dump::<Routing>(&out_dir, "routing");
}

fn dump<T: JsonSchema>(out_dir: &Path, name: &str) {
    let schema = schemars::schema_for!(T);
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
