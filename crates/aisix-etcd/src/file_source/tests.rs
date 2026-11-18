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
"#
    ));
    assert_eq!(
        errors,
        [
            "provider_keys[0]: `id` is required (the resource's UUID)".to_string(),
            "provider_keys[1]: `id` must be a UUID string, found \"openai-prod\"".to_string(),
            "provider_keys[2]: `id` must be a UUID string, found 42".to_string(),
            format!(
                "provider_keys[4]: duplicate id \"{PK}\" — every entry of `provider_keys` needs \
                 its own id"
            ),
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
    let provider = FileConfigProvider::new(&path);
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
    let provider = Arc::new(FileConfigProvider::new(&path));
    let passive = Arc::new(provider.passive());
    let supervisor = Supervisor::with_sources(
        vec![
            (WatchedPrefix::environment(ENV_PREFIX), provider),
            (WatchedPrefix::global(GLOBAL_PREFIX), passive),
        ],
        SnapshotCache::disabled(),
    )
    .with_source_kind(SourceKind::File);

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
