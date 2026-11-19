//! The resources file (`resources_file` in config.yaml) as a configuration
//! channel.
//!
//! A resources file is a set of etcd documents written down: every entry is
//! the `id` of a resource plus the value the control plane would store for
//! it, grouped by the kind segment of its etcd key. Reading the file turns
//! each entry into the [`RawEntry`] etcd would have returned for
//! `<prefix>/<kind>/<id>`, and from there the gateway runs exactly what it
//! runs on etcd: [`crate::loader::build_snapshot`] decodes, validates and
//! assembles the snapshot, and [`crate::Supervisor`] applies it, keeps the
//! last known good value of a row that stops loading, and reports through
//! `/status/config`. So a document loads from a file exactly when it would
//! load from etcd, and a row the loader rejects is dropped and reported
//! instead of failing the gateway.
//!
//! ```yaml
//! _format_version: "2"
//! provider_keys:
//!   - id: 0b6f3c1e-2f0a-4c5e-9d55-7d1a8b0c2e11
//!     display_name: openai-prod
//!     provider: openai
//!     api_key: ${OPENAI_API_KEY}
//! global:            # the shared `<base>/global/` prefix
//!   pricing:
//!     - id: …
//! ```
//!
//! What is this module's own is the channel: reading the file, `${VAR}`
//! interpolation of string scalars ([`yaml`]), and the conversion to keys.
//! A problem there — an unreadable file, invalid YAML, an entry without a
//! UUID `id`, two entries of one kind sharing an id, an unset variable —
//! means the file cannot be turned into a set of etcd rows at all, so it
//! fails the load as a whole: at boot the gateway does not start, and on
//! a SIGHUP reload the previous configuration keeps serving.

mod yaml;

pub use yaml::EnvLookup;

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use yaml_rust2::{Yaml, YamlLoader};

use aisix_core::config_status::{
    hash_entries, ConfigStatus, IncomingRejection, LazyHash, LoadObservation,
};

use crate::key::{PrefixSet, WatchedPrefix};
use crate::provider::{ConfigProvider, ProviderError, RawEntry, WatchEvent};

/// The supported `_format_version`. A missing or different value is a load
/// error, so a file written for another revision of the format is never
/// silently misread.
pub const FORMAT_VERSION: &str = "2";

/// The top-level key holding the documents of the shared global prefix.
pub const GLOBAL_SECTION: &str = "global";

/// The environment prefix the file's documents are keyed under.
pub const ENV_PREFIX: &str = "/aisix";

/// The shared prefix the `global` section's documents are keyed under.
pub const GLOBAL_PREFIX: &str = "/aisix/global/";

/// The prefixes a file's keys resolve against — what
/// [`crate::loader::build_snapshot`] and the supervisor are given in file
/// mode.
pub fn prefixes() -> PrefixSet {
    PrefixSet::new(vec![
        WatchedPrefix::environment(ENV_PREFIX),
        WatchedPrefix::global(GLOBAL_PREFIX),
    ])
}

/// One problem that keeps the file from being read as a set of rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChannelError {
    /// `models[2]`, or `(file)` for a problem with the file as a whole.
    pub scope: String,
    pub message: String,
}

impl std::fmt::Display for ChannelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.scope, self.message)
    }
}

/// Every channel problem in one file, reported together.
#[derive(Debug)]
pub struct FileErrors {
    pub file: String,
    pub errors: Vec<ChannelError>,
    /// Digest of the bytes read, when the file could be read at all.
    pub source_hash: Option<String>,
}

impl std::fmt::Display for FileErrors {
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

impl std::error::Error for FileErrors {}

/// Read `path` into the rows etcd would hold for it, interpolating against
/// the process environment. Every entry is stamped with `revision`.
pub fn read_file(path: &Path, revision: i64) -> Result<Vec<RawEntry>, FileErrors> {
    let label = path.display().to_string();
    let contents = std::fs::read_to_string(path).map_err(|e| FileErrors {
        file: label.clone(),
        errors: vec![file_error(format!("cannot read file: {e}"))],
        source_hash: None,
    })?;
    read_str(&contents, &label, revision, &|name| {
        std::env::var(name).ok()
    })
    .map_err(|mut e| {
        e.source_hash = Some(hash_entries([("", contents.as_bytes())]));
        e
    })
}

fn file_error(message: String) -> ChannelError {
    ChannelError {
        scope: "(file)".into(),
        message,
    }
}

/// [`read_file`] over in-memory contents and a closed environment.
pub fn read_str(
    contents: &str,
    file_label: &str,
    revision: i64,
    env: EnvLookup<'_>,
) -> Result<Vec<RawEntry>, FileErrors> {
    let fail = |errors: Vec<ChannelError>| FileErrors {
        file: file_label.to_string(),
        errors,
        source_hash: None,
    };
    let docs = YamlLoader::load_from_str(contents)
        .map_err(|e| fail(vec![file_error(format!("YAML parse error: {e}"))]))?;
    let root = match docs.as_slice() {
        [] => return Err(fail(vec![file_error("file is empty".into())])),
        [root] => root,
        docs => {
            return Err(fail(vec![file_error(format!(
                "expected a single YAML document, found {}",
                docs.len()
            ))]))
        }
    };
    let Yaml::Hash(root) = root else {
        return Err(fail(vec![file_error(
            "top level must be a mapping with `_format_version` and resource collections".into(),
        )]));
    };

    let mut errors = Vec::new();
    match root.get(&Yaml::String("_format_version".into())) {
        Some(Yaml::String(v)) if v == FORMAT_VERSION => {}
        Some(Yaml::String(v)) => errors.push(file_error(format!(
            "unrecognized _format_version {v:?} (supported: \"{FORMAT_VERSION}\")"
        ))),
        Some(_) => errors.push(file_error(format!(
            "_format_version must be the string \"{FORMAT_VERSION}\" (quote it)"
        ))),
        None => errors.push(file_error(format!(
            "missing mandatory _format_version (expected \"{FORMAT_VERSION}\")"
        ))),
    }

    let mut entries = Vec::new();
    let mut read_section = |section: &yaml_rust2::yaml::Hash, prefix: &str, at: &str| {
        for (kind, items) in section {
            let Yaml::String(kind) = kind else {
                errors.push(file_error(format!(
                    "{at}keys must be plain strings, found {kind:?}"
                )));
                continue;
            };
            if at.is_empty() && (kind == "_format_version" || kind == GLOBAL_SECTION) {
                continue;
            }
            // One key segment: `global/pricing` at the top level would land
            // on the keys of the `global` section's `pricing`.
            if kind.is_empty() || kind.contains('/') {
                errors.push(file_error(format!(
                    "`{at}{kind}` is not a resource collection name (one segment, no `/`)"
                )));
                continue;
            }
            read_kind(
                kind,
                items,
                prefix,
                &format!("{at}{kind}"),
                revision,
                env,
                &mut entries,
                &mut errors,
            );
        }
    };
    read_section(root, ENV_PREFIX, "");
    match root.get(&Yaml::String(GLOBAL_SECTION.into())) {
        None | Some(Yaml::Null) => {}
        Some(Yaml::Hash(global)) => {
            read_section(global, GLOBAL_PREFIX.trim_end_matches('/'), "global.")
        }
        Some(_) => errors.push(file_error(format!(
            "`{GLOBAL_SECTION}` must be a mapping of resource collections"
        ))),
    }

    if errors.is_empty() {
        // Key order, as etcd returns a range: rows sharing a name then
        // resolve the same way from a file as from etcd, and the same way
        // at boot and on every reload.
        entries.sort_by(|a, b| a.key.cmp(&b.key));
        Ok(entries)
    } else {
        Err(fail(errors))
    }
}

#[allow(clippy::too_many_arguments)]
fn read_kind(
    kind: &str,
    items: &Yaml,
    prefix: &str,
    scope: &str,
    revision: i64,
    env: EnvLookup<'_>,
    entries: &mut Vec<RawEntry>,
    errors: &mut Vec<ChannelError>,
) {
    let items = match items {
        Yaml::Null => return,
        Yaml::Array(items) => items,
        _ => {
            errors.push(file_error(format!("`{scope}` must be a sequence of maps")));
            return;
        }
    };
    let mut seen = BTreeSet::new();
    for (i, item) in items.iter().enumerate() {
        let mut error = |message: String| {
            errors.push(ChannelError {
                scope: format!("{scope}[{i}]"),
                message,
            })
        };
        if !matches!(item, Yaml::Hash(_)) {
            error("entry must be a mapping".into());
            continue;
        }
        let mut doc = match yaml::yaml_to_json(item, "", env) {
            Ok(Value::Object(doc)) => doc,
            Ok(_) => unreachable!("a YAML mapping converts to a JSON object"),
            Err((path, message)) => {
                error(format!("field `{path}`: {message}"));
                continue;
            }
        };
        // Only the canonical spelling: references match ids as strings, so
        // `ABC…` or `{abc…}` would name a different resource than `abc…`.
        let id = match doc.remove("id") {
            Some(Value::String(id))
                if uuid::Uuid::try_parse(&id).is_ok_and(|u| u.hyphenated().to_string() == id) =>
            {
                id
            }
            // The value is not echoed: after interpolation it can be anything,
            // a credential included.
            Some(_) => {
                error("`id` must be a UUID string in lowercase hyphenated form".into());
                continue;
            }
            None => {
                error("`id` is required (the resource's UUID)".into());
                continue;
            }
        };
        if !seen.insert(id.clone()) {
            error(format!(
                "duplicate id {id:?} — every entry of `{scope}` needs its own id"
            ));
            continue;
        }
        entries.push(RawEntry {
            key: format!("{prefix}/{kind}/{id}"),
            value: serde_json::to_vec(&Value::Object(doc)).expect("a JSON object serializes"),
            revision,
        });
    }
}

/// One read of the file: its rows and the generation they were read at.
type Read = (Vec<RawEntry>, i64);

/// A [`ConfigProvider`] over a resources file, so the file runs through the
/// same [`crate::Supervisor`] etcd does.
///
/// The file has no watch: a SIGHUP re-reads it and hands the supervisor a
/// full resync. A reload that cannot be read as rows is not handed over at
/// all: the configuration that is serving stays, the failure is recorded on
/// the status handle given by [`Self::with_status`] (`out_of_sync`, with the
/// problems as `rejected[]`), and the next SIGHUP tries again.
///
/// The file is read only at boot and on SIGHUP. A `load_all` after the
/// first successful read — the supervisor's own first cycle right after
/// the boot apply, or a cycle it restarts — hands back the rows last read
/// rather than reading again, so nothing a file is changed to applies
/// without a SIGHUP, and that cycle cannot fail on a file being edited.
///
/// Both of the supervisor's prefixes are read from the one file, so only
/// the environment provider reads it; the provider for the global prefix
/// is [`Self::passive`], which hands back the global rows of that read.
/// They have to come from it: the supervisor takes each key from the
/// read of the prefix that owns it. Every resync
/// carries the whole file, both prefixes included, because a resync
/// replaces everything the supervisor holds.
#[derive(Clone)]
pub struct FileConfigProvider {
    path: Option<PathBuf>,
    generation: Arc<AtomicI64>,
    /// The rows of the last successful read, and their generation.
    last_read: Arc<std::sync::Mutex<Option<Read>>>,
    status: Option<ConfigStatus>,
    /// The SIGHUP receiver, installed when the provider is built so a
    /// signal sent after the first apply (readiness) but before the watch
    /// starts is queued for it rather than taking the default action,
    /// which ends the process.
    #[cfg(unix)]
    hup: Arc<std::sync::Mutex<Option<tokio::signal::unix::Signal>>>,
}

impl FileConfigProvider {
    /// Inside a Tokio runtime on unix this installs the SIGHUP receiver,
    /// and fails if it cannot; outside one, [`ConfigProvider::watch`]
    /// installs it instead. The runtime must have its I/O driver enabled
    /// (the gateway's does), which signal handling needs.
    pub fn new(path: impl Into<PathBuf>) -> std::io::Result<Self> {
        #[cfg(unix)]
        let hup = match tokio::runtime::Handle::try_current() {
            Ok(_) => Some(tokio::signal::unix::signal(
                tokio::signal::unix::SignalKind::hangup(),
            )?),
            Err(_) => None,
        };
        Ok(Self {
            path: Some(path.into()),
            generation: Arc::default(),
            last_read: Arc::default(),
            status: None,
            #[cfg(unix)]
            hup: Arc::new(std::sync::Mutex::new(hup)),
        })
    }

    /// Where a failed SIGHUP reload is reported: the supervisor's own
    /// status handle (see [`crate::Supervisor::with_config_status`]).
    pub fn with_status(mut self, status: ConfigStatus) -> Self {
        self.status = Some(status);
        self
    }

    /// The provider for the global prefix: its rows are the global rows of
    /// this provider's last read, at the generation they were read at.
    pub fn passive(&self) -> Self {
        Self {
            path: None,
            generation: Arc::clone(&self.generation),
            last_read: Arc::clone(&self.last_read),
            status: None,
            #[cfg(unix)]
            hup: Arc::default(),
        }
    }

    /// Read the file under the next generation, which is the revision its
    /// rows carry.
    pub fn read(&self) -> Result<(Vec<RawEntry>, i64), FileErrors> {
        let Some(path) = &self.path else {
            return Ok((Vec::new(), self.generation.load(Ordering::Relaxed)));
        };
        let revision = self.generation.fetch_add(1, Ordering::Relaxed) + 1;
        read_file(path, revision).map(|entries| (entries, revision))
    }

    async fn read_blocking(&self) -> Result<(Vec<RawEntry>, i64), FileErrors> {
        let this = self.clone();
        let label = self
            .path
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_default();
        let result = tokio::task::spawn_blocking(move || this.read())
            .await
            .unwrap_or_else(|e| {
                Err(FileErrors {
                    file: label,
                    errors: vec![file_error(format!("read task failed: {e}"))],
                    source_hash: None,
                })
            });
        match &result {
            Ok(read) if self.path.is_some() => {
                *self.last_read.lock().unwrap_or_else(|p| p.into_inner()) = Some(read.clone());
            }
            Ok(_) => {}
            Err(report) => tracing::warn!("resources file could not be read:\n{report}"),
        }
        result
    }

    fn report_failure(&self, report: &FileErrors) {
        let Some(status) = &self.status else { return };
        let seen_at = chrono::Utc::now();
        let rejected = report
            .errors
            .iter()
            .enumerate()
            // Several problems can share a scope (`(file)`); each is its own row.
            .map(|(i, e)| IncomingRejection {
                identity: format!("{} #{i}", e.scope),
                resource_kind: e
                    .scope
                    .split('[')
                    .next()
                    .filter(|k| !k.starts_with('('))
                    .unwrap_or_default()
                    .to_string(),
                resource_id: String::new(),
                last_error_kind: if e.message.starts_with("`id`")
                    || e.message.starts_with("duplicate id")
                {
                    "bad_key"
                } else {
                    "non_json"
                }
                .to_string(),
                last_error: e.message.clone(),
                seen_at,
                serving_stale_since: None,
            })
            .collect();
        status.record_load(LoadObservation {
            source_hash: LazyHash::ready(
                report
                    .source_hash
                    .clone()
                    .unwrap_or_else(|| hash_entries(std::iter::empty())),
            ),
            observed_revision: None,
            applied: None,
            rejected,
            partially_compatible: Vec::new(),
            partially_compatible_rows_by_kind: Default::default(),
            stale_served_rows_by_kind: Default::default(),
            is_reload: true,
            wholly_rejected: true,
        });
    }
}

#[async_trait]
impl ConfigProvider for FileConfigProvider {
    async fn load_all(&self) -> Result<(Vec<RawEntry>, i64), ProviderError> {
        if let Some(last) = self
            .last_read
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
        {
            if self.path.is_none() {
                let (entries, revision) = last;
                let global = entries
                    .into_iter()
                    .filter(|e| e.key.starts_with(GLOBAL_PREFIX))
                    .collect();
                return Ok((global, revision));
            }
            return Ok(last);
        }
        self.read_blocking()
            .await
            .map_err(|report| ProviderError::File(report.to_string()))
    }

    async fn watch(
        &self,
        _start_revision: i64,
    ) -> Result<
        Box<dyn futures::Stream<Item = Result<WatchEvent, ProviderError>> + Send + Unpin>,
        ProviderError,
    > {
        if self.path.is_none() {
            return Ok(Box::new(futures::stream::pending()));
        }
        #[cfg(unix)]
        {
            use tokio::signal::unix::{signal, SignalKind};
            let installed = self.hup.lock().unwrap_or_else(|p| p.into_inner()).take();
            let hup = match installed {
                Some(hup) => hup,
                None => signal(SignalKind::hangup()).map_err(|e| {
                    ProviderError::Watch(format!("cannot install the SIGHUP handler: {e}"))
                })?,
            };
            let reader = self.clone();
            let stream = futures::stream::unfold((hup, reader), |(mut hup, reader)| async move {
                loop {
                    hup.recv().await?;
                    match reader.read_blocking().await {
                        Ok((entries, revision)) => {
                            let event = WatchEvent::Resync {
                                entries: Arc::new(entries),
                                revision,
                            };
                            return Some((Ok(event), (hup, reader)));
                        }
                        Err(report) => reader.report_failure(&report),
                    }
                }
            });
            Ok(Box::new(Box::pin(stream)))
        }
        #[cfg(not(unix))]
        Ok(Box::new(futures::stream::pending()))
    }
}

#[cfg(test)]
mod tests;
