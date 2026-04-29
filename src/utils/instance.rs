use std::{path::Path, sync::OnceLock};

use anyhow::{Result, anyhow};
use log::{debug, info, warn};
use uuid::Uuid;

const ID_FILE: &str = "/tmp/aisix_instance_id";

static RUN_ID: OnceLock<String> = OnceLock::new();
static INSTANCE_ID: OnceLock<String> = OnceLock::new();

/// Initialize the instance ID and run ID.
pub fn init() -> Result<()> {
    RUN_ID.get_or_init(|| Uuid::new_v4().to_string());

    let instance_id = resolve_instance_id_from_path(Path::new(ID_FILE))?;
    INSTANCE_ID.get_or_init(|| instance_id);

    Ok(())
}

/// Get the run ID.
pub fn run_id() -> String {
    RUN_ID.get().cloned().expect("run id has been initialized")
}

/// Get the instance ID.
pub fn instance_id() -> String {
    INSTANCE_ID
        .get()
        .cloned()
        .expect("instance id has been initialized")
}

fn resolve_instance_id_from_path(path: &Path) -> Result<String> {
    if let Ok(Some(id)) = read_id_file(path) {
        debug!("agent: loaded instance_id from {:?}", path);
        return Ok(id);
    }

    let id = Uuid::new_v4().to_string();

    if let Ok(()) = write_id_file(path, &id) {
        info!(
            "agent: generated and persisted instance_id={id} to {:?}",
            path
        );
        return Ok(id);
    }

    warn!(
        "agent: instance_id={id} is in-memory only (could not write to {:?}) — identifier will rotate on restart",
        path
    );
    Ok(id)
}

fn read_id_file(path: &Path) -> Result<Option<String>> {
    match std::fs::read_to_string(path) {
        Ok(content) => {
            let trimmed = content.trim();
            if !trimmed.is_empty() {
                return Ok(Some(trimmed.to_string()));
            }
            warn!("agent: instance_id file {:?} is empty, ignoring", path);
            Ok(None)
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(anyhow!("failed to read instance_id file {path:?}: {e}")),
    }
}

fn write_id_file(path: &Path, id: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        if let Err(e) = std::fs::create_dir_all(parent) {
            debug!(
                "agent: cannot create instance_id parent dir {:?}: {e}",
                parent
            );
            return Err(anyhow!(
                "failed to create instance_id parent dir {parent:?}: {e}"
            ));
        }
    }

    if let Err(e) = std::fs::write(path, id) {
        debug!("agent: cannot write instance_id to {:?}: {e}", path);
        return Err(anyhow!("failed to write instance_id file {path:?}: {e}"));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    fn tmp_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("aisix_instance_test_{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn reuses_existing_file() {
        let dir = tmp_dir();
        let path = dir.join("instance_id");
        std::fs::write(&path, "existing-id\n").unwrap();

        assert_eq!(resolve_instance_id_from_path(&path).unwrap(), "existing-id");
    }

    #[test]
    fn writes_new_id_when_file_absent() {
        let dir = tmp_dir();
        let path = dir.join("sub/instance_id");

        let id = resolve_instance_id_from_path(&path).unwrap();
        assert!(!id.is_empty());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), id);
    }

    #[test]
    fn falls_back_to_memory_when_unwritable() {
        let dir = tmp_dir();
        let blocked = dir.join("blocked");
        std::fs::write(&blocked, "not-a-dir").unwrap();

        let target = blocked.join("instance_id");
        let id = resolve_instance_id_from_path(&target).unwrap();
        assert!(!id.is_empty());
        assert!(!target.exists());
    }
}
