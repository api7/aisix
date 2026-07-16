//! Load-observability bridge for the file source.
//!
//! Turns a file load outcome ([`FileSourceErrors`] or a snapshot) into a
//! [`LoadObservation`] and records it on a [`ConfigStatus`], so the standalone
//! file mode reports the same `/status/config` shape as the etcd path. The
//! file source is all-or-nothing: a clean load applies the whole file, a
//! failing reload keeps the last-good snapshot (flagged `wholly_rejected`).

use std::path::Path;

use chrono::Utc;

use crate::config_status::{
    hash_bytes, AppliedSnapshot, ConfigStatus, IncomingRejection, LoadObservation,
};
use crate::AisixSnapshot;

use super::{load_from_str, FileSourceErrors, LoadError};

/// Load `path` and record the outcome on `config_status`, returning the same
/// `Result` as [`super::load_resources_file`]. `is_reload` counts the load
/// toward `aisix_config_reloads_total` (boot and SIGHUP reloads both do).
///
/// A file that cannot be read is a fetch failure (source unreachable); a file
/// that loads with errors is a wholesale rejection (last-good retained).
pub fn load_resources_file_tracked(
    path: &Path,
    revision: i64,
    is_reload: bool,
    config_status: &ConfigStatus,
) -> Result<AisixSnapshot, FileSourceErrors> {
    let label = path.display().to_string();
    let contents = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            // Cannot even read the file: a fetch failure, not a per-resource
            // rejection. Leave the last-good applied snapshot intact.
            config_status.record_fetch_failure();
            return Err(FileSourceErrors {
                file: label,
                errors: vec![LoadError {
                    scope: "(file)".into(),
                    message: format!("cannot read file: {e}"),
                }],
            });
        }
    };

    let source_hash = hash_bytes(contents.as_bytes());
    let result = load_from_str(&contents, &label, revision, &|name| {
        std::env::var(name).ok()
    });

    match &result {
        Ok(snapshot) => {
            config_status.record_load(LoadObservation {
                source_hash: source_hash.clone(),
                observed_revision: None,
                applied: Some(AppliedSnapshot {
                    // All-or-nothing: the applied set is the whole file, so the
                    // applied hash equals the source hash on a clean load.
                    config_hash: source_hash,
                    revision: None,
                    resource_counts: resource_counts(snapshot),
                }),
                rejected: vec![],
                is_reload,
                wholly_rejected: false,
            });
        }
        Err(errors) => {
            let now = Utc::now();
            let rejected = errors
                .errors
                .iter()
                .map(|e| map_load_error(e, now))
                .collect();
            config_status.record_load(LoadObservation {
                source_hash,
                observed_revision: None,
                // No snapshot applied — the previous one keeps serving.
                applied: None,
                rejected,
                is_reload,
                // The whole file was rejected; last-good retained.
                wholly_rejected: true,
            });
        }
    }

    result
}

/// Per-kind counts of the loaded snapshot, keyed by the plural resource kind.
fn resource_counts(snap: &AisixSnapshot) -> std::collections::BTreeMap<String, usize> {
    let mut counts = std::collections::BTreeMap::new();
    for (kind, n) in [
        ("models", snap.models.len()),
        ("api_keys", snap.apikeys.len()),
        ("provider_keys", snap.provider_keys.len()),
        ("guardrails", snap.guardrails.len()),
        ("guardrail_attachments", snap.guardrail_attachments.len()),
        ("cache_policies", snap.cache_policies.len()),
        (
            "observability_exporters",
            snap.observability_exporters.len(),
        ),
        ("rate_limit_policies", snap.rate_limit_policies.len()),
        ("mcp_servers", snap.mcp_servers.len()),
        ("a2a_agents", snap.a2a_agents.len()),
    ] {
        if n > 0 {
            counts.insert(kind.to_string(), n);
        }
    }
    counts
}

/// Map a file [`LoadError`] to the source-agnostic rejection wire shape.
///
/// The file source carries no per-entry `RejectionKind`, so the kind is
/// classified from the message: a YAML parse failure is `non_json` (a source-
/// format problem), a canonical-decode failure is `parse_failed`, and every
/// other validation / reference / structure error is `schema_failed`. The
/// scope (`models[2] ("gpt-4o")`) yields `resource_kind` / `resource_id`;
/// a file-level scope (`(file)`) leaves both empty, mirroring the etcd path.
fn map_load_error(e: &LoadError, seen_at: chrono::DateTime<Utc>) -> IncomingRejection {
    let (resource_kind, resource_id) = parse_scope(&e.scope);
    IncomingRejection {
        identity: e.scope.clone(),
        resource_kind,
        resource_id,
        last_error_kind: classify(&e.message).to_string(),
        last_error: e.message.clone(),
        seen_at,
    }
}

fn classify(message: &str) -> &'static str {
    if message.contains("YAML parse error") {
        "non_json"
    } else if message.starts_with("cannot decode canonical document") {
        "parse_failed"
    } else {
        "schema_failed"
    }
}

/// Split a file scope into `(kind, id)`. `models[2] ("gpt-4o")` → `("models",
/// "gpt-4o")`; `models[2]` → `("models", "")`; `(file)` → `("", "")`.
fn parse_scope(scope: &str) -> (String, String) {
    if scope.starts_with('(') {
        return (String::new(), String::new());
    }
    let kind = scope
        .split(['[', ' '])
        .next()
        .unwrap_or("")
        .trim()
        .to_string();
    let id = scope
        .split_once('"')
        .and_then(|(_, rest)| rest.split_once('"'))
        .map(|(id, _)| id.to_string())
        .unwrap_or_default();
    (kind, id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config_status::{ConfigState, SourceKind};

    #[test]
    fn parse_scope_extracts_kind_and_id() {
        assert_eq!(
            parse_scope("models[2] (\"gpt-4o\")"),
            ("models".to_string(), "gpt-4o".to_string())
        );
        assert_eq!(
            parse_scope("provider_keys[0]"),
            ("provider_keys".to_string(), String::new())
        );
        assert_eq!(parse_scope("(file)"), (String::new(), String::new()));
    }

    #[test]
    fn classify_maps_message_to_error_kind() {
        assert_eq!(classify("YAML parse error: ..."), "non_json");
        assert_eq!(
            classify("cannot decode canonical document: missing field"),
            "parse_failed"
        );
        assert_eq!(
            classify("schema validation failed at `/name`"),
            "schema_failed"
        );
    }

    #[test]
    fn tracked_load_of_valid_file_reports_synced_without_revisions() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("resources.yaml");
        std::fs::write(
            &path,
            "_format_version: \"1\"\nprovider_keys:\n  - display_name: pk\n    api_key: sk-x\n",
        )
        .unwrap();
        let cs = ConfigStatus::new(SourceKind::File);
        let result = load_resources_file_tracked(&path, 1, true, &cs);
        assert!(result.is_ok(), "load failed: {:?}", result.err());
        let view = cs.view();
        assert_eq!(view.state, ConfigState::Synced);
        // File mode omits revision fields.
        let json = serde_json::to_value(&view).unwrap();
        assert!(json["source"].get("observed_revision").is_none());
        assert!(json["applied"].get("applied_revision").is_none());
        assert_eq!(json["applied"]["resource_counts"]["provider_keys"], 1);
    }

    #[test]
    fn tracked_reload_failure_keeps_last_good_and_reports_out_of_sync() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("resources.yaml");
        std::fs::write(
            &path,
            "_format_version: \"1\"\nprovider_keys:\n  - display_name: pk\n    api_key: sk-x\n",
        )
        .unwrap();
        let cs = ConfigStatus::new(SourceKind::File);
        load_resources_file_tracked(&path, 1, true, &cs).unwrap();

        // A broken reload: unknown top-level key.
        std::fs::write(&path, "_format_version: \"1\"\nnot_a_collection: []\n").unwrap();
        let result = load_resources_file_tracked(&path, 2, true, &cs);
        assert!(result.is_err());
        assert_eq!(cs.view().state, ConfigState::OutOfSync);
        assert!(!cs.view().rejected.is_empty());
    }
}
