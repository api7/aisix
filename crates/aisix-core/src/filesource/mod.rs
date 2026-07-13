//! Standalone file-based resource source (`resources_file` in
//! config.yaml).
//!
//! Loads every dynamic resource — provider keys, models, API keys,
//! guardrails, MCP servers, A2A agents, cache policies, observability
//! exporters, rate-limit policies — from one YAML file instead of etcd,
//! so a single container can run fully declaratively.
//!
//! Pipeline (identical for boot, SIGHUP reload, and `aisix validate`):
//!
//! ```text
//! read file → YAML parse → ${VAR} interpolation (string scalars)
//!   → per-entry JSON documents → file sugar desugared
//!   → canonical JSON-Schema validation (same validators as the etcd path)
//!   → typed serde models → cross-reference checks → AisixSnapshot
//! ```
//!
//! Errors are collected across the whole file — every load problem is
//! reported together with kind / entry / field context, instead of
//! failing on the first one.
//!
//! File format v1 (`_format_version: "1"`, mandatory):
//! - nine top-level collection keys, each a sequence of maps, named by
//!   the plural resource kind; unknown top-level keys are load errors.
//! - after desugaring, every entry must be exactly a canonical resource
//!   document (`schemas/resources/*.schema.json`) — the file source
//!   never relaxes the canonical schemas.
//! - entries carry no `id`: ids are derived deterministically
//!   (UUIDv5 of `"<kind>/<identity>"`, see
//!   [`desugar::FILE_RESOURCE_NAMESPACE`]) and identities must be
//!   unique per kind.

mod desugar;
mod yaml;

pub use desugar::{derive_id, FILE_RESOURCE_NAMESPACE};

use serde_json::Value;
use std::collections::BTreeMap;
use std::path::Path;
use yaml_rust2::{Yaml, YamlLoader};

use crate::models::{
    validate_a2a_agent, validate_apikey, validate_cache_policy, validate_guardrail,
    validate_mcp_server, validate_model, validate_observability_exporter, validate_provider_key,
    validate_rate_limit_policy, A2aAgent, ApiKey, CachePolicy, Guardrail, McpServer, Model,
    ObservabilityExporter, ProviderKey, RateLimitPolicy, SchemaError,
};
use crate::resource::ResourceEntry;
use crate::AisixSnapshot;

use desugar::{IdentityField, IdentityMaps};
use yaml::EnvLookup;

/// One load problem, with enough context to fix it: the entry scope
/// (`models[2] ("gpt-4o")`, or `(file)` for file-level problems) and a
/// message that names the offending field where applicable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadError {
    pub scope: String,
    pub message: String,
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.scope, self.message)
    }
}

/// Aggregated load failure: every error found across the whole file.
#[derive(Debug)]
pub struct FileSourceErrors {
    /// The file the errors refer to, as given by the caller.
    pub file: String,
    pub errors: Vec<LoadError>,
}

impl std::fmt::Display for FileSourceErrors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "resources file {}: {} error(s):",
            self.file,
            self.errors.len()
        )?;
        for e in &self.errors {
            writeln!(f, "  - {e}")?;
        }
        Ok(())
    }
}

impl std::error::Error for FileSourceErrors {}

/// The supported format version. A missing or unrecognized
/// `_format_version` is a load error, so future format revisions can
/// change semantics without silently misreading old gateways' files.
const SUPPORTED_FORMAT_VERSION: &str = "1";

/// Fixed processing order for the nine resource collections.
const KINDS: [(&str, IdentityField); 9] = [
    ("provider_keys", IdentityField::DisplayName),
    ("models", IdentityField::DisplayName),
    ("api_keys", IdentityField::DisplayName),
    ("guardrails", IdentityField::Name),
    ("mcp_servers", IdentityField::NameOrDisplayName),
    ("a2a_agents", IdentityField::NameOrDisplayName),
    ("cache_policies", IdentityField::Name),
    ("observability_exporters", IdentityField::Name),
    ("rate_limit_policies", IdentityField::Name),
];

/// Load `path` into a fresh [`AisixSnapshot`], resolving `${VAR}`
/// interpolation against the current process environment. `revision` is
/// stamped on every entry (the file source's generation counter: 1 at
/// boot, incremented per successful reload).
pub fn load_resources_file(path: &Path, revision: i64) -> Result<AisixSnapshot, FileSourceErrors> {
    let label = path.display().to_string();
    let contents = std::fs::read_to_string(path).map_err(|e| FileSourceErrors {
        file: label.clone(),
        errors: vec![LoadError {
            scope: "(file)".into(),
            message: format!("cannot read file: {e}"),
        }],
    })?;
    load_from_str(&contents, &label, revision, &|name| {
        std::env::var(name).ok()
    })
}

/// The full pipeline over in-memory contents. Separated from
/// [`load_resources_file`] so tests can inject file contents and a
/// closed environment map without touching process state.
pub fn load_from_str(
    contents: &str,
    file_label: &str,
    revision: i64,
    env: EnvLookup<'_>,
) -> Result<AisixSnapshot, FileSourceErrors> {
    let mut errors: Vec<LoadError> = Vec::new();
    let fail = |errors: Vec<LoadError>| FileSourceErrors {
        file: file_label.to_string(),
        errors,
    };
    let file_error = |message: String| LoadError {
        scope: "(file)".into(),
        message,
    };

    // ── YAML parse ────────────────────────────────────────────────────
    let docs = match YamlLoader::load_from_str(contents) {
        Ok(docs) => docs,
        Err(e) => return Err(fail(vec![file_error(format!("YAML parse error: {e}"))])),
    };
    let root = match docs.len() {
        0 => return Err(fail(vec![file_error("file is empty".into())])),
        1 => &docs[0],
        n => {
            return Err(fail(vec![file_error(format!(
                "expected a single YAML document, found {n}"
            ))]))
        }
    };
    let Yaml::Hash(root_map) = root else {
        return Err(fail(vec![file_error(
            "top level must be a mapping with `_format_version` and resource collections".into(),
        )]));
    };

    // ── Format-version gate + unknown-top-level-key gate ─────────────
    match root_map.get(&Yaml::String("_format_version".into())) {
        Some(Yaml::String(v)) if v == SUPPORTED_FORMAT_VERSION => {}
        Some(Yaml::String(v)) => errors.push(file_error(format!(
            "unrecognized _format_version {v:?} (supported: \"{SUPPORTED_FORMAT_VERSION}\")"
        ))),
        Some(_) => errors.push(file_error(format!(
            "_format_version must be the string \"{SUPPORTED_FORMAT_VERSION}\" (quote it)"
        ))),
        None => errors.push(file_error(format!(
            "missing mandatory _format_version (expected \"{SUPPORTED_FORMAT_VERSION}\")"
        ))),
    }
    for key in root_map.keys() {
        let Yaml::String(key) = key else {
            errors.push(file_error(format!(
                "top-level keys must be plain strings, found {key:?}"
            )));
            continue;
        };
        if key != "_format_version" && !KINDS.iter().any(|(k, _)| k == key) {
            let known: Vec<&str> = KINDS.iter().map(|(k, _)| *k).collect();
            errors.push(file_error(format!(
                "unknown top-level key `{key}` (expected _format_version and resource \
                 collections: {})",
                known.join(", ")
            )));
        }
    }

    // ── Pass 1: interpolate + convert + identity / duplicate checks ──
    struct Prepared {
        kind: &'static str,
        scope: String,
        identity: String,
        doc: Value,
    }
    let mut prepared: Vec<Prepared> = Vec::new();
    let mut identity_maps = IdentityMaps::new();

    for (kind, identity_field) in KINDS {
        let mut seen_at: BTreeMap<String, usize> = BTreeMap::new();
        let entries = match root_map.get(&Yaml::String(kind.into())) {
            None | Some(Yaml::Null) => continue,
            Some(Yaml::Array(items)) => items,
            Some(_) => {
                errors.push(file_error(format!("`{kind}` must be a sequence of maps")));
                continue;
            }
        };
        for (i, item) in entries.iter().enumerate() {
            let index_scope = format!("{kind}[{i}]");
            if !matches!(item, Yaml::Hash(_)) {
                errors.push(LoadError {
                    scope: index_scope,
                    message: "entry must be a mapping".into(),
                });
                continue;
            }
            let doc = match yaml::yaml_to_json(item, "", env) {
                Ok(doc) => doc,
                Err((path, message)) => {
                    errors.push(LoadError {
                        scope: index_scope,
                        message: format!("field `{path}`: {message}"),
                    });
                    continue;
                }
            };
            let Some(identity) = identity_field.extract(&doc) else {
                errors.push(LoadError {
                    scope: index_scope,
                    message: format!(
                        "{} is required and must be a non-empty string \
                         (it is the entry's identity)",
                        identity_field.describe()
                    ),
                });
                continue;
            };
            let scope = format!("{kind}[{i}] ({identity:?})");
            if let Some(first) = seen_at.get(&identity) {
                errors.push(LoadError {
                    scope,
                    message: format!(
                        "duplicate {kind} entry: {} {identity:?} is already \
                         defined at {kind}[{first}] — identities must be unique \
                         within a kind",
                        identity_field.describe()
                    ),
                });
                continue;
            }
            seen_at.insert(identity.clone(), i);
            identity_maps
                .entry(kind)
                .or_default()
                .insert(identity.clone(), derive_id(kind, &identity));
            prepared.push(Prepared {
                kind,
                scope,
                identity,
                doc,
            });
        }
    }

    // ── Pass 2: desugar → canonical validation → typed models ────────
    fn finish<T: serde::de::DeserializeOwned>(
        scope: &str,
        doc: &Value,
        validate: fn(&Value) -> Result<(), SchemaError>,
        errors: &mut Vec<LoadError>,
    ) -> Option<T> {
        if let Err(e) = validate(doc) {
            errors.push(LoadError {
                scope: scope.to_string(),
                message: e.to_string(),
            });
            return None;
        }
        match serde_json::from_value::<T>(doc.clone()) {
            Ok(t) => Some(t),
            Err(e) => {
                errors.push(LoadError {
                    scope: scope.to_string(),
                    message: format!("cannot decode canonical document: {e}"),
                });
                None
            }
        }
    }

    // Typed buckets carry `(derived_id, scope, value)` so cross-reference
    // errors can point back at the file entry.
    let mut models: Vec<(String, String, Model)> = Vec::new();
    let mut apikeys: Vec<(String, String, ApiKey)> = Vec::new();
    let mut provider_keys: Vec<(String, String, ProviderKey)> = Vec::new();
    let mut guardrails: Vec<(String, String, Guardrail)> = Vec::new();
    let mut mcp_servers: Vec<(String, String, McpServer)> = Vec::new();
    let mut a2a_agents: Vec<(String, String, A2aAgent)> = Vec::new();
    let mut cache_policies: Vec<(String, String, CachePolicy)> = Vec::new();
    let mut observability_exporters: Vec<(String, String, ObservabilityExporter)> = Vec::new();
    let mut rate_limit_policies: Vec<(String, String, RateLimitPolicy)> = Vec::new();

    for mut entry in prepared {
        let id = derive_id(entry.kind, &entry.identity);
        let scope = entry.scope;

        // The file accepts no `id` field on any kind: ids are always
        // derived. Strict schemas would reject it as an unknown field
        // anyway; the open ones (guardrail, cache_policy,
        // observability_exporter) would silently carry it, so the check
        // is made explicit and uniform here.
        if entry.doc.get("id").is_some() {
            errors.push(LoadError {
                scope,
                message: "the resources file does not accept `id` — ids are derived \
                          deterministically from the entry's name"
                    .into(),
            });
            continue;
        }

        let sugar_result = match entry.kind {
            "models" => desugar::desugar_model(&mut entry.doc, &identity_maps),
            "api_keys" => desugar::desugar_api_key(&mut entry.doc, env),
            "rate_limit_policies" => {
                desugar::desugar_rate_limit_policy(&mut entry.doc, &identity_maps)
            }
            _ => Ok(()),
        };
        if let Err(message) = sugar_result {
            errors.push(LoadError { scope, message });
            continue;
        }

        match entry.kind {
            "provider_keys" => {
                if let Some(t) = finish(&scope, &entry.doc, validate_provider_key, &mut errors) {
                    provider_keys.push((id, scope, t));
                }
            }
            "models" => {
                if let Some(t) = finish(&scope, &entry.doc, validate_model, &mut errors) {
                    models.push((id, scope, t));
                }
            }
            "api_keys" => {
                if let Some(t) = finish(&scope, &entry.doc, validate_apikey, &mut errors) {
                    apikeys.push((id, scope, t));
                }
            }
            "guardrails" => {
                if let Some(t) = finish(&scope, &entry.doc, validate_guardrail, &mut errors) {
                    guardrails.push((id, scope, t));
                }
            }
            "mcp_servers" => {
                if let Some(t) = finish(&scope, &entry.doc, validate_mcp_server, &mut errors) {
                    mcp_servers.push((id, scope, t));
                }
            }
            "a2a_agents" => {
                if let Some(t) = finish(&scope, &entry.doc, validate_a2a_agent, &mut errors) {
                    a2a_agents.push((id, scope, t));
                }
            }
            "cache_policies" => {
                if let Some(t) = finish(&scope, &entry.doc, validate_cache_policy, &mut errors) {
                    cache_policies.push((id, scope, t));
                }
            }
            "observability_exporters" => {
                if let Some(t) = finish(
                    &scope,
                    &entry.doc,
                    validate_observability_exporter,
                    &mut errors,
                ) {
                    observability_exporters.push((id, scope, t));
                }
            }
            "rate_limit_policies" => {
                if let Some(t) = finish(&scope, &entry.doc, validate_rate_limit_policy, &mut errors)
                {
                    rate_limit_policies.push((id, scope, t));
                }
            }
            other => unreachable!("kind {other} is not in KINDS"),
        }
    }

    // ── Pass 3: cross-reference checks ────────────────────────────────
    // Every model-name reference must resolve at load time so a typo can
    // never become a silent runtime failure. Entries containing `*` are
    // glob patterns and are exempt from the existence check.
    let empty = BTreeMap::new();
    let model_names = identity_maps.get("models").unwrap_or(&empty);
    let mut check_model_ref = |scope: &str, field: &str, reference: &str| {
        if reference.contains('*') || model_names.contains_key(reference) {
            return;
        }
        let mut known: Vec<&str> = model_names.keys().map(String::as_str).collect();
        known.sort_unstable();
        errors.push(LoadError {
            scope: scope.to_string(),
            message: format!(
                "{field} references unknown model {reference:?} (defined models: {})",
                if known.is_empty() {
                    "none".to_string()
                } else {
                    known.join(", ")
                }
            ),
        });
    };

    for (_, scope, key) in &apikeys {
        for entry in &key.allowed_models {
            check_model_ref(scope, "allowed_models entry", entry);
        }
    }
    for (_, scope, model) in &models {
        if let Some(routing) = &model.routing {
            for target in &routing.targets {
                check_model_ref(scope, "routing target", &target.model);
            }
        }
        if let Some(ensemble) = &model.ensemble {
            for member in &ensemble.panel {
                check_model_ref(scope, "ensemble panel member", &member.model);
            }
            check_model_ref(scope, "ensemble judge", &ensemble.judge.model);
        }
        if let Some(semantic) = &model.semantic {
            check_model_ref(scope, "semantic embedding_model", &semantic.embedding_model);
            check_model_ref(scope, "semantic default", &semantic.default);
            for route in &semantic.routes {
                check_model_ref(
                    scope,
                    &format!("semantic route {:?} target", route.name),
                    &route.target,
                );
            }
            if let crate::models::OnEmbeddingFailure::Target { target } =
                &semantic.on_embedding_failure
            {
                check_model_ref(scope, "semantic on_embedding_failure target", target);
            }
        }
    }

    if !errors.is_empty() {
        return Err(fail(errors));
    }

    // ── Materialize the snapshot ──────────────────────────────────────
    let snapshot = AisixSnapshot::new();
    for (id, _, v) in provider_keys {
        snapshot
            .provider_keys
            .insert(ResourceEntry::new(id, v, revision));
    }
    for (id, _, v) in models {
        snapshot.models.insert(ResourceEntry::new(id, v, revision));
    }
    for (id, _, v) in apikeys {
        snapshot.apikeys.insert(ResourceEntry::new(id, v, revision));
    }
    for (id, _, v) in guardrails {
        snapshot
            .guardrails
            .insert(ResourceEntry::new(id, v, revision));
    }
    for (id, _, v) in mcp_servers {
        snapshot
            .mcp_servers
            .insert(ResourceEntry::new(id, v, revision));
    }
    for (id, _, v) in a2a_agents {
        snapshot
            .a2a_agents
            .insert(ResourceEntry::new(id, v, revision));
    }
    for (id, _, v) in cache_policies {
        snapshot
            .cache_policies
            .insert(ResourceEntry::new(id, v, revision));
    }
    for (id, _, v) in observability_exporters {
        snapshot
            .observability_exporters
            .insert(ResourceEntry::new(id, v, revision));
    }
    for (id, _, v) in rate_limit_policies {
        snapshot
            .rate_limit_policies
            .insert(ResourceEntry::new(id, v, revision));
    }
    Ok(snapshot)
}

#[cfg(test)]
mod tests;
