//! Memory readings taken at scrape time: the allocator's own counters, the
//! process as the kernel sees it, the memory limit it runs under, the
//! async runtimes' task counts, and how much each in-process store holds.
//!
//! Everything here is read when Prometheus asks, never on the request
//! path. Each reading is O(1) or bounded by something small and fixed
//! (a file under `/proc`, a runtime count, a shard count) — a probe that
//! would have to walk a store is not added at all.

use std::sync::{Arc, Mutex};

/// One in-process store's current size.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ComponentReading {
    /// `None` for a store measured only in bytes.
    pub entries: Option<u64>,
    /// Exact bytes held, only where the store already accounts them.
    pub bytes: Option<u64>,
}

type ComponentProbe = Box<dyn Fn() -> ComponentReading + Send + Sync>;
type ExporterQueuesProbe = Box<dyn Fn() -> Vec<(String, u64)> + Send + Sync>;

/// What the scrape reads beyond the recorder: component stores,
/// registered by whoever owns them. Cheap to clone; every clone
/// sees the same registrations.
#[derive(Clone, Default)]
pub struct MemoryProbes {
    inner: Arc<Mutex<ProbeSet>>,
}

#[derive(Default)]
struct ProbeSet {
    components: Vec<(&'static str, ComponentProbe)>,
    exporter_queues: Option<ExporterQueuesProbe>,
}

impl std::fmt::Debug for MemoryProbes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryProbes").finish_non_exhaustive()
    }
}

impl MemoryProbes {
    pub fn new() -> Self {
        Self::default()
    }

    /// Report a store under `component = name`. `read` runs on every
    /// scrape, off the async runtime, and must be O(1).
    pub fn register_component(
        &self,
        name: &'static str,
        read: impl Fn() -> ComponentReading + Send + Sync + 'static,
    ) {
        self.inner
            .lock()
            .expect("memory probes")
            .components
            .push((name, Box::new(read)));
    }

    /// Report each exporter's queue depth under `component =
    /// exporter_queue`, labelled by exporter name.
    pub fn register_exporter_queues(
        &self,
        read: impl Fn() -> Vec<(String, u64)> + Send + Sync + 'static,
    ) {
        self.inner.lock().expect("memory probes").exporter_queues = Some(Box::new(read));
    }

    pub(crate) fn read_exporter_queues(&self) -> Vec<(String, u64)> {
        let set = self.inner.lock().expect("memory probes");
        set.exporter_queues
            .as_ref()
            .map_or_else(Vec::new, |read| read())
    }

    /// Readings summed per component name: a store that exists once per
    /// worker registers once per worker and reports as one.
    pub(crate) fn read_components(&self) -> Vec<(&'static str, ComponentReading)> {
        let set = self.inner.lock().expect("memory probes");
        let mut out: Vec<(&'static str, ComponentReading)> = Vec::new();
        for (name, read) in &set.components {
            let reading = read();
            match out.iter_mut().find(|(n, _)| n == name) {
                Some((_, sum)) => {
                    sum.entries = add(sum.entries, reading.entries);
                    sum.bytes = add(sum.bytes, reading.bytes);
                }
                None => out.push((name, reading)),
            }
        }
        out
    }
}

fn add(a: Option<u64>, b: Option<u64>) -> Option<u64> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a + b),
        (a, b) => a.or(b),
    }
}

/// Every async runtime in the process, by name. Process-wide rather than
/// per [`MemoryProbes`] because a thread-per-core worker builds its runtime
/// deep inside the listener, far from any handle to the metrics.
static RUNTIMES: Mutex<Vec<(String, tokio::runtime::Handle)>> = Mutex::new(Vec::new());

/// Report `handle`'s task counts under `runtime = name`.
pub fn register_runtime(name: impl Into<String>, handle: tokio::runtime::Handle) {
    RUNTIMES
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .push((name.into(), handle));
}

pub(crate) fn for_each_runtime(mut visit: impl FnMut(&str, &tokio::runtime::Handle)) {
    let runtimes = RUNTIMES.lock().unwrap_or_else(|p| p.into_inner());
    for (name, handle) in runtimes.iter() {
        visit(name, handle);
    }
}

/// The allocator's own byte counts, after advancing its statistics epoch
/// so they are current. `None` where jemalloc is not the allocator.
pub fn allocator_stats() -> Option<[(&'static str, u64); 6]> {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    {
        use tikv_jemalloc_ctl::{epoch, stats};
        epoch::advance().ok()?;
        Some([
            ("allocated", stats::allocated::read().ok()? as u64),
            ("active", stats::active::read().ok()? as u64),
            ("resident", stats::resident::read().ok()? as u64),
            ("mapped", stats::mapped::read().ok()? as u64),
            ("retained", stats::retained::read().ok()? as u64),
            ("metadata", stats::metadata::read().ok()? as u64),
        ])
    }
    #[cfg(not(all(target_os = "linux", target_env = "gnu")))]
    {
        None
    }
}

/// The process as the kernel accounts it, in the units the standard
/// Prometheus `process_*` families use.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProcessStats {
    pub resident_bytes: u64,
    pub virtual_bytes: u64,
    pub threads: u64,
    pub open_fds: u64,
    pub max_fds: u64,
    pub cpu_seconds: f64,
    pub start_time_seconds: f64,
}

#[cfg(target_os = "linux")]
pub fn process_stats() -> Option<ProcessStats> {
    let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
    // The command name is parenthesised and may itself contain spaces or
    // parentheses, so fields are counted from the LAST `)`.
    let rest = &stat[stat.rfind(')')? + 2..];
    let fields: Vec<&str> = rest.split_whitespace().collect();
    // `rest` starts at field 3 (state); field N sits at index N - 3.
    let field = |n: usize| fields.get(n - 3).and_then(|v| v.parse::<u64>().ok());
    let ticks = rustix::param::clock_ticks_per_second() as f64;
    let page = rustix::param::page_size() as u64;
    let boot_time = std::fs::read_to_string("/proc/stat")
        .ok()?
        .lines()
        .find_map(|line| line.strip_prefix("btime "))
        .and_then(|v| v.trim().parse::<u64>().ok())?;
    let open_fds = std::fs::read_dir("/proc/self/fd").ok()?.count() as u64;
    let max_fds = rustix::process::getrlimit(rustix::process::Resource::Nofile)
        .current
        .unwrap_or(u64::MAX);
    Some(ProcessStats {
        resident_bytes: field(24)? * page,
        virtual_bytes: field(23)?,
        threads: field(20)?,
        open_fds,
        max_fds,
        cpu_seconds: (field(14)? + field(15)?) as f64 / ticks,
        start_time_seconds: boot_time as f64 + field(22)? as f64 / ticks,
    })
}

#[cfg(not(target_os = "linux"))]
pub fn process_stats() -> Option<ProcessStats> {
    None
}

/// Resident set size alone — what the auto-dump check reads every second,
/// so it avoids the directory walk and the second file `process_stats`
/// needs.
#[cfg(target_os = "linux")]
pub fn resident_bytes() -> Option<u64> {
    let statm = std::fs::read_to_string("/proc/self/statm").ok()?;
    let pages: u64 = statm.split_whitespace().nth(1)?.parse().ok()?;
    Some(pages * rustix::param::page_size() as u64)
}

#[cfg(not(target_os = "linux"))]
pub fn resident_bytes() -> Option<u64> {
    None
}

/// The cgroup memory limit this process runs under: v2 `memory.max`, or
/// v1 `memory.limit_in_bytes`. `None` when there is none — an unlimited
/// cgroup, or no cgroup filesystem at all.
///
/// Read fresh each time rather than cached: a pod's limit can be resized
/// in place.
pub fn memory_limit_bytes() -> Option<u64> {
    let cgroups = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    memory_limit_from(&cgroups, |path| std::fs::read_to_string(path).ok())
}

/// A v1 cgroup reports "no limit" as the largest page-aligned `i64`, not
/// as a word; anything this large is not a limit anyone set.
const V1_UNLIMITED_FLOOR: u64 = 1 << 62;

fn memory_limit_from(cgroups: &str, read: impl Fn(&str) -> Option<String>) -> Option<u64> {
    for line in cgroups.lines() {
        let mut parts = line.splitn(3, ':');
        let (Some(id), Some(controllers), Some(path)) = (parts.next(), parts.next(), parts.next())
        else {
            continue;
        };
        let (candidates, v1) = if id == "0" && controllers.is_empty() {
            (
                [
                    format!("/sys/fs/cgroup{path}/memory.max"),
                    "/sys/fs/cgroup/memory.max".to_string(),
                ],
                false,
            )
        } else if controllers.split(',').any(|c| c == "memory") {
            (
                [
                    format!("/sys/fs/cgroup/memory{path}/memory.limit_in_bytes"),
                    "/sys/fs/cgroup/memory/memory.limit_in_bytes".to_string(),
                ],
                true,
            )
        } else {
            continue;
        };
        // Inside a container the cgroup namespace usually makes the
        // listed path `/`, but without one the host path is listed and the
        // container sees its own cgroup at the mount root instead.
        for candidate in &candidates {
            let Some(raw) = read(candidate) else {
                continue;
            };
            let raw = raw.trim();
            if raw == "max" {
                return None;
            }
            return match raw.parse::<u64>() {
                Ok(limit) if v1 && limit >= V1_UNLIMITED_FLOOR => None,
                Ok(limit) => Some(limit),
                Err(_) => None,
            };
        }
    }
    None
}

/// The host's total memory, the auto-dump fallback denominator when the
/// process has no cgroup limit.
pub fn host_memory_bytes() -> Option<u64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    meminfo
        .lines()
        .find_map(|line| line.strip_prefix("MemTotal:"))
        .and_then(|rest| {
            rest.trim()
                .trim_end_matches("kB")
                .trim()
                .parse::<u64>()
                .ok()
        })
        .map(|kib| kib * 1024)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn files<'a>(entries: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |path| {
            entries
                .iter()
                .find(|(p, _)| *p == path)
                .map(|(_, v)| v.to_string())
        }
    }

    #[test]
    fn cgroup_v2_limit_reads_the_process_cgroup_then_the_mount_root() {
        let own = files(&[("/sys/fs/cgroup/kubepods/pod1/memory.max", "536870912\n")]);
        assert_eq!(
            memory_limit_from("0::/kubepods/pod1\n", own),
            Some(536870912)
        );
        let root = files(&[("/sys/fs/cgroup/memory.max", "1073741824\n")]);
        assert_eq!(
            memory_limit_from("0::/kubepods/pod1\n", root),
            Some(1073741824)
        );
        let unlimited = files(&[("/sys/fs/cgroup/memory.max", "max\n")]);
        assert_eq!(memory_limit_from("0::/\n", unlimited), None);
    }

    #[test]
    fn cgroup_v1_limit_treats_the_page_aligned_maximum_as_no_limit() {
        let cgroups = "12:cpu,cpuacct:/docker/x\n9:memory:/docker/x\n";
        let set = files(&[(
            "/sys/fs/cgroup/memory/docker/x/memory.limit_in_bytes",
            "268435456\n",
        )]);
        assert_eq!(memory_limit_from(cgroups, set), Some(268435456));
        let unset = files(&[(
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
            "9223372036854771712\n",
        )]);
        assert_eq!(memory_limit_from(cgroups, unset), None);
    }

    #[test]
    fn no_cgroup_filesystem_means_no_limit() {
        assert_eq!(memory_limit_from("0::/\n", files(&[])), None);
        assert_eq!(memory_limit_from("", files(&[])), None);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn process_stats_describe_this_process() {
        let stats = process_stats().expect("linux exposes /proc/self");
        assert!(stats.resident_bytes > 0);
        assert!(stats.virtual_bytes >= stats.resident_bytes);
        assert!(stats.threads >= 1);
        assert!(stats.open_fds >= 1 && stats.open_fds <= stats.max_fds);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();
        assert!(stats.start_time_seconds > now - 86_400.0 && stats.start_time_seconds <= now + 1.0);
        assert!(resident_bytes().unwrap() > 0);
    }

    #[test]
    fn component_readings_sum_per_name() {
        let probes = MemoryProbes::new();
        probes.register_component("a", || ComponentReading {
            entries: Some(2),
            bytes: Some(10),
        });
        probes.register_component("a", || ComponentReading {
            entries: Some(3),
            bytes: Some(5),
        });
        probes.register_component("b", || ComponentReading {
            entries: Some(1),
            bytes: None,
        });
        let read = probes.read_components();
        assert_eq!(
            read,
            vec![
                (
                    "a",
                    ComponentReading {
                        entries: Some(5),
                        bytes: Some(15)
                    }
                ),
                (
                    "b",
                    ComponentReading {
                        entries: Some(1),
                        bytes: None
                    }
                ),
            ]
        );
    }
}
