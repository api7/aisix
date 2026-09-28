//! Heap profiles from jemalloc's sampling profiler, rendered as gzipped
//! pprof with function names resolved in-process.
//!
//! Sampling is compiled into the shipped Linux build and switched on by the
//! binary's embedded allocator configuration, so a profile can be taken
//! from a gateway that is already misbehaving — no restart, no rebuild.
//! Two things take one: `GET /debug/pprof/heap` on the debug listener, and
//! [`spawn_auto_dump`], which writes one to disk as resident memory nears
//! the limit, so the evidence outlives the OOM kill that usually follows.
//!
//! Names are resolved from the binary's symbol table, which the release
//! build keeps; it carries no DWARF, so frames have function names but no
//! file or line.

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;

use aisix_core::AutoDumpConfig;

use crate::metrics::Metrics;

#[derive(Debug, thiserror::Error)]
pub enum DumpError {
    #[error("heap profiling is not available in this build")]
    Unsupported,
    #[error("heap profiling is not enabled (opt.prof is false)")]
    NotEnabled,
    #[error("a heap profile is already being taken")]
    Busy,
    #[error("taking the heap profile failed: {0}")]
    Failed(String),
}

/// One dump at a time. A dump walks every live sample and symbolizes it,
/// which is seconds of CPU on a large heap; two at once only double that.
static DUMP_LOCK: Mutex<()> = Mutex::new(());

/// Whether this process can produce a heap profile at all.
pub fn profiling_enabled() -> bool {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    {
        // SAFETY: `opt.prof` is a read-only boolean mallctl.
        unsafe { tikv_jemalloc_ctl::raw::read::<bool>(b"opt.prof\0") }.unwrap_or(false)
    }
    #[cfg(not(all(target_os = "linux", target_env = "gnu")))]
    {
        false
    }
}

/// Take a heap profile now, returning the gzipped pprof bytes.
///
/// Blocking and CPU-heavy: call it off the async runtime. With `wait`
/// false a dump already in progress is reported as [`DumpError::Busy`];
/// with `wait` true this queues behind it. jemalloc can only dump to a
/// file, so the raw profile passes through `scratch_dir` and is removed
/// before this returns.
pub fn dump_pprof(scratch_dir: &Path, wait: bool) -> Result<Vec<u8>, DumpError> {
    let _guard = if wait {
        DUMP_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    } else {
        match DUMP_LOCK.try_lock() {
            Ok(guard) => guard,
            Err(std::sync::TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
            Err(std::sync::TryLockError::WouldBlock) => return Err(DumpError::Busy),
        }
    };
    dump_locked(scratch_dir)
}

#[cfg(all(target_os = "linux", target_env = "gnu"))]
fn dump_locked(scratch_dir: &Path) -> Result<Vec<u8>, DumpError> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);

    if !profiling_enabled() {
        return Err(DumpError::NotEnabled);
    }
    let path = scratch_dir.join(format!(
        ".aisix-heap-{}-{}.tmp",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    let c_path = std::ffi::CString::new(path.as_os_str().as_encoded_bytes())
        .map_err(|e| DumpError::Failed(e.to_string()))?;
    // SAFETY: `prof.dump` takes a NUL-terminated path that outlives the call.
    unsafe { tikv_jemalloc_ctl::raw::write(b"prof.dump\0", c_path.as_ptr()) }
        .map_err(|e| DumpError::Failed(format!("prof.dump to {}: {e}", path.display())))?;
    let file = std::fs::File::open(&path);
    let _ = std::fs::remove_file(&path);
    let file = file.map_err(|e| DumpError::Failed(e.to_string()))?;
    let profile =
        pprof_util::parse_jeheap(std::io::BufReader::new(file), mappings::MAPPINGS.as_deref())
            .map_err(|e| DumpError::Failed(e.to_string()))?;
    let pprof = profile.to_pprof(("inuse_space", "bytes"), ("space", "bytes"), None);
    // Symbolizing parses the binary's symbol table and keeps it cached for
    // the life of the process. A dump is rare and often taken near the
    // memory limit, so give that memory back rather than hold it forever.
    backtrace::clear_symbol_cache();
    Ok(pprof)
}

#[cfg(not(all(target_os = "linux", target_env = "gnu")))]
fn dump_locked(_scratch_dir: &Path) -> Result<Vec<u8>, DumpError> {
    Err(DumpError::Unsupported)
}

/// Where the raw profile passes through on its way to pprof: the auto-dump
/// directory when it is usable — under a read-only root filesystem it is
/// the one writable path — and the system temp directory otherwise.
pub fn scratch_dir(auto_dump: &AutoDumpConfig) -> PathBuf {
    let dir = Path::new(&auto_dump.dir);
    if dir_is_writable(dir) {
        dir.to_path_buf()
    } else {
        std::env::temp_dir()
    }
}

fn dir_is_writable(dir: &Path) -> bool {
    if !dir.is_dir() {
        return false;
    }
    let probe = dir.join(format!(".aisix-write-probe-{}", std::process::id()));
    let ok = std::fs::write(&probe, b"").is_ok();
    let _ = std::fs::remove_file(&probe);
    ok
}

/// A threshold re-arms once resident memory has fallen this far below it,
/// so a process hovering at a threshold dumps once, not every second.
const REARM_GAP: f64 = 0.05;

/// Threshold edge detection for the auto-dump check.
#[derive(Debug)]
struct Arming {
    thresholds: Vec<f64>,
    armed: Vec<bool>,
}

impl Arming {
    fn new(mut thresholds: Vec<f64>) -> Self {
        thresholds.sort_by(f64::total_cmp);
        thresholds.dedup();
        let armed = vec![true; thresholds.len()];
        Self { thresholds, armed }
    }

    /// The thresholds `fraction` has just crossed upward; each fires once
    /// and stays disarmed until the fraction falls [`REARM_GAP`] below it.
    fn observe(&mut self, fraction: f64) -> Vec<f64> {
        let mut fired = Vec::new();
        for (threshold, armed) in self.thresholds.iter().zip(self.armed.iter_mut()) {
            if *armed && fraction >= *threshold {
                *armed = false;
                fired.push(*threshold);
            } else if !*armed && fraction <= threshold - REARM_GAP {
                *armed = true;
            }
        }
        fired
    }
}

/// A threshold as a percentage for a file name: `0.8` is `80`, `0.00001`
/// is `0.001`.
fn percent_label(threshold: f64) -> String {
    let pct = threshold * 100.0;
    if (pct - pct.round()).abs() < 1e-9 {
        format!("{}", pct.round() as u64)
    } else {
        let text = format!("{pct:.6}");
        text.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

fn is_auto_dump(name: &str) -> bool {
    name.contains("-auto-") && name.ends_with(".pb.gz") && !name.starts_with('.')
}

/// Delete the oldest auto dumps beyond `keep`. Names start with a UTC
/// timestamp, so name order is age order.
fn prune(dir: &Path, keep: usize) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut names: Vec<String> = entries
        .filter_map(|e| e.ok()?.file_name().into_string().ok())
        .filter(|name| is_auto_dump(name))
        .collect();
    names.sort();
    let excess = names.len().saturating_sub(keep);
    for name in &names[..excess] {
        if let Err(error) = std::fs::remove_file(dir.join(name)) {
            tracing::warn!(%error, file = %name, "could not delete an old heap profile");
        }
    }
}

fn write_dump(dir: &Path, threshold: f64, pprof: &[u8]) -> std::io::Result<PathBuf> {
    let name = format!(
        "{}-auto-{}.pb.gz",
        chrono::Utc::now().format("%Y%m%dT%H%M%S%.3fZ"),
        percent_label(threshold)
    );
    let path = dir.join(&name);
    let tmp = dir.join(format!(".{name}.tmp"));
    std::fs::write(&tmp, pprof)?;
    std::fs::rename(&tmp, &path).inspect_err(|_| {
        let _ = std::fs::remove_file(&tmp);
    })?;
    Ok(path)
}

/// Resident memory as a fraction of what the process may use: the cgroup
/// limit, or the host's memory when there is none.
fn memory_fraction() -> Option<f64> {
    let rss = crate::memory::resident_bytes()?;
    let limit = crate::memory::memory_limit_bytes().or_else(crate::memory::host_memory_bytes)?;
    (limit > 0).then(|| rss as f64 / limit as f64)
}

const CHECK_INTERVAL: Duration = Duration::from_secs(1);

/// Start the auto-dump check, or explain once why it cannot run.
pub fn spawn_auto_dump(cfg: &AutoDumpConfig, metrics: Metrics) {
    if !cfg.enabled || cfg.thresholds.is_empty() {
        return;
    }
    if !profiling_enabled() {
        tracing::info!(
            "heap profiling is not active in this process; automatic heap dumps are off"
        );
        return;
    }
    let dir = PathBuf::from(&cfg.dir);
    if !dir_is_writable(&dir) {
        tracing::warn!(
            dir = %dir.display(),
            "observability.heap_profiling.auto_dump.dir is missing or not writable; \
             automatic heap dumps are off"
        );
        return;
    }
    let keep = cfg.keep;
    let mut arming = Arming::new(cfg.thresholds.clone());
    let spawned = std::thread::Builder::new()
        .name("heap-autodump".into())
        .spawn(move || {
            // Dumping and symbolizing is seconds of CPU on a large heap,
            // taken exactly when the process is already under pressure;
            // the request workers come first.
            aisix_core::sched::demote_current_thread();
            loop {
                std::thread::sleep(CHECK_INTERVAL);
                let Some(fraction) = memory_fraction() else {
                    continue;
                };
                for threshold in arming.observe(fraction) {
                    let written = dump_pprof(&dir, true)
                        .map_err(|e| e.to_string())
                        .and_then(|pprof| {
                            write_dump(&dir, threshold, &pprof).map_err(|e| e.to_string())
                        });
                    match written {
                        Ok(path) => {
                            metrics.record_heap_profile_dump("auto", true);
                            tracing::warn!(
                                file = %path.display(),
                                threshold,
                                fraction,
                                "resident memory crossed a heap-dump threshold; heap profile written"
                            );
                            prune(&dir, keep);
                        }
                        Err(error) => {
                            metrics.record_heap_profile_dump("auto", false);
                            tracing::error!(%error, threshold, "automatic heap dump failed");
                        }
                    }
                }
            }
        });
    if let Err(error) = spawned {
        tracing::warn!(%error, "could not start the automatic heap dump thread");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_threshold_fires_once_per_upward_crossing() {
        let mut arming = Arming::new(vec![0.9, 0.8]);
        assert!(arming.observe(0.5).is_empty());
        assert_eq!(arming.observe(0.85), vec![0.8]);
        assert!(arming.observe(0.86).is_empty(), "no refire while above");
        assert_eq!(arming.observe(0.95), vec![0.9]);
        // 0.78 is within the re-arm gap of 0.8: still disarmed.
        assert!(arming.observe(0.78).is_empty());
        assert!(arming.observe(0.84).is_empty());
        // 0.7 re-arms 0.8 (0.9 re-armed at 0.78), so the next rise fires both.
        assert!(arming.observe(0.7).is_empty());
        assert_eq!(arming.observe(0.95), vec![0.8, 0.9]);
    }

    #[test]
    fn percent_labels() {
        assert_eq!(percent_label(0.8), "80");
        assert_eq!(percent_label(1.0), "100");
        assert_eq!(percent_label(0.00001), "0.001");
        assert_eq!(percent_label(0.125), "12.5");
    }

    #[test]
    fn prune_keeps_the_newest_auto_dumps_only() {
        let dir = tempfile::tempdir().unwrap();
        for name in [
            "20260101T000000.000Z-auto-80.pb.gz",
            "20260101T000001.000Z-auto-90.pb.gz",
            "20260101T000002.000Z-auto-80.pb.gz",
            "unrelated.pb.gz",
            ".20260101T000003.000Z-auto-80.pb.gz.tmp",
        ] {
            std::fs::write(dir.path().join(name), b"x").unwrap();
        }
        prune(dir.path(), 2);
        let mut left: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        left.sort();
        assert_eq!(
            left,
            vec![
                ".20260101T000003.000Z-auto-80.pb.gz.tmp",
                "20260101T000001.000Z-auto-90.pb.gz",
                "20260101T000002.000Z-auto-80.pb.gz",
                "unrelated.pb.gz",
            ]
        );
    }

    #[test]
    fn scratch_falls_back_to_temp_when_the_dump_dir_is_unusable() {
        let cfg = AutoDumpConfig {
            dir: "/nonexistent/aisix-heap".into(),
            ..AutoDumpConfig::default()
        };
        assert_eq!(scratch_dir(&cfg), std::env::temp_dir());
        let dir = tempfile::tempdir().unwrap();
        let cfg = AutoDumpConfig {
            dir: dir.path().to_string_lossy().into_owned(),
            ..AutoDumpConfig::default()
        };
        assert_eq!(scratch_dir(&cfg), dir.path());
    }
}
