use super::*;
use crate::loader::{build_snapshot, RejectionKind};

const PK: &str = "0b6f3c1e-2f0a-4c5e-9d55-7d1a8b0c2e11";
const MODEL: &str = "6efd592d-a73b-573d-8e2c-e9e103692f92";

fn env(name: &str) -> Option<String> {
    (name == "OPENAI_API_KEY").then(|| "sk-from-env".to_string())
}

fn read(contents: &str) -> Result<Vec<RawEntry>, FileErrors> {
    read_str(contents, "resources.yaml", 7, &env)
}

fn messages(contents: &str) -> Vec<String> {
    read(contents)
        .expect_err("the file must not read")
        .errors
        .iter()
        .map(ToString::to_string)
        .collect()
}

#[test]
fn entries_become_the_rows_etcd_would_hold() {
    let entries = read(&format!(
        r#"_format_version: "2"
provider_keys:
  - id: {PK}
    display_name: openai
    provider: openai
    api_key: ${{OPENAI_API_KEY}}
global:
  pricing:
    - id: {MODEL}
      key: gpt-4o
      input_per_1k: 1
      output_per_1k: 2
"#
    ))
    .unwrap();
    let keys: Vec<&str> = entries.iter().map(|e| e.key.as_str()).collect();
    assert_eq!(
        keys,
        [
            format!("/aisix/provider_keys/{PK}"),
            format!("/aisix/global/pricing/{MODEL}"),
        ]
    );
    let value: Value = serde_json::from_slice(&entries[0].value).unwrap();
    assert_eq!(
        value,
        serde_json::json!({"display_name": "openai", "provider": "openai", "api_key": "sk-from-env"}),
        "`id` is the key, not part of the value, and `${{VAR}}` is interpolated"
    );
    assert!(entries.iter().all(|e| e.revision == 7));

    let (snap, stats) = build_snapshot(&prefixes(), &entries);
    assert_eq!(stats.accepted, 2, "rejections: {:?}", stats.rejections);
    assert_eq!(
        snap.provider_keys.get_by_id(PK).unwrap().value.api_key,
        "sk-from-env"
    );
    assert_eq!(snap.global_pricing.len(), 1);
}

#[test]
fn a_row_the_loader_rejects_is_dropped_and_the_rest_still_load() {
    let entries = read(&format!(
        r#"_format_version: "2"
provider_keys:
  - id: {PK}
    display_name: openai
    api_key: sk-x
models:
  - id: {MODEL}
    display_name: 42
widgets:
  - id: {MODEL}
"#
    ))
    .unwrap();
    let (snap, stats) = build_snapshot(&prefixes(), &entries);
    assert_eq!(stats.accepted, 1);
    assert!(snap.provider_keys.get_by_id(PK).is_some());
    let kinds: Vec<(String, RejectionKind)> = stats
        .rejections
        .iter()
        .map(|r| (r.key.clone(), r.kind))
        .collect();
    assert_eq!(
        kinds,
        [
            (
                format!("/aisix/models/{MODEL}"),
                RejectionKind::SchemaFailed
            ),
            (
                format!("/aisix/widgets/{MODEL}"),
                RejectionKind::UnknownKind
            ),
        ]
    );
}

#[test]
fn an_entry_without_a_uuid_id_fails_the_file() {
    let errors = messages(&format!(
        r#"_format_version: "2"
provider_keys:
  - display_name: no-id
  - id: openai-prod
  - id: 42
  - id: {PK}
  - id: {PK}
  - id: 0B6F3C1E-2F0A-4C5E-9D55-7D1A8B0C2E11
  - id: "{{0b6f3c1e-2f0a-4c5e-9d55-7d1a8b0c2e11}}"
"#
    ));
    let not_canonical = |i: usize| {
        format!("provider_keys[{i}]: `id` must be a UUID string in lowercase hyphenated form")
    };
    assert_eq!(
        errors,
        [
            "provider_keys[0]: `id` is required (the resource's UUID)".to_string(),
            not_canonical(1),
            not_canonical(2),
            format!(
                "provider_keys[4]: duplicate id \"{PK}\" — every entry of `provider_keys` needs \
                 its own id"
            ),
            not_canonical(5),
            not_canonical(6),
        ]
    );
}

#[test]
fn one_id_may_be_used_by_two_kinds() {
    let entries = read(&format!(
        r#"_format_version: "2"
provider_keys:
  - id: {PK}
models:
  - id: {PK}
"#
    ))
    .unwrap();
    assert_eq!(entries.len(), 2);
}

#[test]
fn channel_problems_are_reported_together() {
    let errors = messages(
        r#"_format_version: "1"
models:
  - id: 6efd592d-a73b-573d-8e2c-e9e103692f92
    api_base: ${UNSET}
  - just-a-string
api_keys: {}
"#,
    );
    assert_eq!(
        errors,
        [
            "(file): unrecognized _format_version \"1\" (supported: \"2\")",
            "models[0]: field `api_base`: environment variable `UNSET` is unset or empty",
            "models[1]: entry must be a mapping",
            "(file): `api_keys` must be a sequence of maps",
        ]
    );
}

#[test]
fn unparseable_or_empty_files_fail() {
    assert!(messages("a: [")[0].starts_with("(file): YAML parse error"));
    assert_eq!(messages(""), ["(file): file is empty"]);
    assert_eq!(
        messages("models: []"),
        ["(file): missing mandatory _format_version (expected \"2\")"]
    );
}

#[test]
fn an_unreadable_file_fails() {
    let err = read_file(Path::new("/nonexistent/resources.yaml"), 1).unwrap_err();
    assert!(
        err.errors[0].message.starts_with("cannot read file"),
        "{err}"
    );
}

#[tokio::test]
async fn the_passive_provider_reports_the_generation_of_the_last_read() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("resources.yaml");
    std::fs::write(&path, "_format_version: \"2\"\n").unwrap();
    let provider = FileConfigProvider::new(&path).unwrap();
    let passive = provider.passive();
    assert_eq!(provider.load_all().await.unwrap().1, 1);
    assert_eq!(provider.load_all().await.unwrap().1, 2);
    assert_eq!(passive.load_all().await.unwrap(), (Vec::new(), 2));
}

#[tokio::test]
async fn a_file_runs_through_the_supervisor_like_etcd() {
    use crate::{SnapshotCache, Supervisor};
    use aisix_core::config_status::{ConfigState, SourceKind};

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("resources.yaml");
    let write = |model_name: &str| {
        std::fs::write(
            &path,
            format!(
                "_format_version: \"2\"\nprovider_keys:\n  - id: {PK}\n    display_name: openai\n    \
                 api_key: sk-x\nmodels:\n  - id: {MODEL}\n    display_name: {model_name}\n    \
                 provider: openai\n    model_name: gpt-4o\n    provider_key_id: {PK}\n"
            ),
        )
        .unwrap()
    };
    write("gpt-4o");
    let status = aisix_core::ConfigStatus::new(SourceKind::File);
    let provider = Arc::new(
        FileConfigProvider::new(&path)
            .unwrap()
            .with_status(status.clone()),
    );
    let passive = Arc::new(provider.passive());
    let supervisor = Supervisor::with_sources(
        vec![
            (WatchedPrefix::environment(ENV_PREFIX), provider),
            (WatchedPrefix::global(GLOBAL_PREFIX), passive),
        ],
        SnapshotCache::disabled(),
    )
    .with_config_status(status.clone());

    supervisor.load_once().await.unwrap();
    let snap = supervisor.handle().load();
    assert_eq!(
        snap.models.get_by_id(MODEL).unwrap().value.display_name,
        "gpt-4o"
    );
    let view = supervisor.config_status().view();
    assert_eq!(view.state, ConfigState::Synced);
    assert_eq!(
        serde_json::to_value(&view).unwrap()["source"]["type"],
        "file"
    );

    // A row that stops loading keeps serving its last known good value
    // and is reported, as it would be from etcd.
    write("[not, a, string]");
    supervisor.load_once().await.unwrap();
    let snap = supervisor.handle().load();
    assert_eq!(
        snap.models.get_by_id(MODEL).unwrap().value.display_name,
        "gpt-4o"
    );
    let view = supervisor.config_status().view();
    assert_eq!(view.state, ConfigState::Degraded);
    let rejected = serde_json::to_value(&view).unwrap()["rejected"].clone();
    assert_eq!(rejected[0]["resource_kind"], "models");
    assert_eq!(rejected[0]["resource_id"], MODEL);
}

#[tokio::test]
async fn a_reload_that_cannot_be_read_is_reported_out_of_sync() {
    use crate::{SnapshotCache, Supervisor};
    use aisix_core::config_status::{ConfigState, SourceKind};

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("resources.yaml");
    std::fs::write(
        &path,
        format!("_format_version: \"2\"\nprovider_keys:\n  - id: {PK}\n    display_name: pk\n    api_key: x\n"),
    )
    .unwrap();
    let status = aisix_core::ConfigStatus::new(SourceKind::File);
    let provider = Arc::new(
        FileConfigProvider::new(&path)
            .unwrap()
            .with_status(status.clone()),
    );
    let supervisor = Supervisor::with_sources(
        vec![(
            WatchedPrefix::environment(ENV_PREFIX),
            Arc::clone(&provider),
        )],
        SnapshotCache::disabled(),
    )
    .with_config_status(status.clone());
    supervisor.load_once().await.unwrap();
    assert_eq!(status.view().state, ConfigState::Synced);

    std::fs::write(
        &path,
        "_format_version: \"2\"\nmodels:\n  - display_name: no-id\n",
    )
    .unwrap();
    let report = provider.read().unwrap_err();
    provider.report_failure(&report);
    let view = serde_json::to_value(status.view()).unwrap();
    assert_eq!(status.view().state, ConfigState::OutOfSync);
    assert_eq!(view["rejected"][0]["resource_kind"], "models");
    assert_eq!(view["rejected"][0]["last_error_kind"], "bad_key");
    // The configuration that was serving still serves.
    assert!(supervisor
        .handle()
        .load()
        .provider_keys
        .get_by_id(PK)
        .is_some());
}

#[test]
fn a_collection_name_is_one_key_segment_and_an_id_is_never_echoed() {
    let errors = messages(
        "_format_version: \"2\"\nglobal/pricing: []\nmodels:\n  - id: ${OPENAI_API_KEY}\n",
    );
    assert_eq!(
        errors,
        [
            "(file): `global/pricing` is not a resource collection name (one segment, no `/`)",
            "models[0]: `id` must be a UUID string in lowercase hyphenated form",
        ]
    );
}

#[tokio::test]
async fn a_file_gone_after_the_first_apply_leaves_that_apply_serving() {
    use crate::{SnapshotCache, Supervisor};
    use aisix_core::config_status::{ConfigState, SourceKind};

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("resources.yaml");
    std::fs::write(
        &path,
        format!("_format_version: \"2\"\nprovider_keys:\n  - id: {PK}\n    display_name: pk\n    api_key: x\n"),
    )
    .unwrap();
    let status = aisix_core::ConfigStatus::new(SourceKind::File);
    let provider = Arc::new(
        FileConfigProvider::new(&path)
            .unwrap()
            .with_status(status.clone()),
    );
    let supervisor = Supervisor::with_sources(
        vec![(WatchedPrefix::environment(ENV_PREFIX), provider)],
        SnapshotCache::disabled(),
    )
    .with_config_status(status.clone());
    supervisor.load_once().await.unwrap();

    std::fs::remove_file(&path).unwrap();
    assert!(supervisor.load_once().await.is_err());
    assert!(
        status.is_ready(),
        "the first apply opened readiness and nothing closes it"
    );
    assert!(supervisor
        .handle()
        .load()
        .provider_keys
        .get_by_id(PK)
        .is_some());
    assert_ne!(status.view().state, ConfigState::NeverLoaded);
}

/// A SIGHUP that arrives after the provider exists but before its watch
/// starts — the window between the boot apply opening readiness and the
/// supervisor reaching `watch` — is queued for the watch, not lost and not
/// the default action that would end the process.
#[cfg(unix)]
#[tokio::test]
async fn a_sighup_before_the_watch_starts_is_delivered_to_it() {
    use futures::StreamExt;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("resources.yaml");
    std::fs::write(&path, "_format_version: \"2\"\n").unwrap();
    let provider = FileConfigProvider::new(&path).unwrap();
    let status = std::process::Command::new("kill")
        .args(["-HUP", &std::process::id().to_string()])
        .status()
        .unwrap();
    assert!(status.success());
    let mut stream = provider.watch(0).await.unwrap();
    let event = tokio::time::timeout(std::time::Duration::from_secs(5), stream.next())
        .await
        .expect("the queued SIGHUP reaches the watch")
        .expect("the stream is open")
        .expect("the file reads");
    assert!(matches!(event, WatchEvent::Resync { .. }));
}
