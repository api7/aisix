//! The production logging path against a consumer that stops reading.
//!
//! This is the defect in its original shape: a child process installs the
//! real subscriber, its stderr is a pipe, and the parent does not read
//! that pipe. The pipe fills, the sink blocks, and the child keeps
//! logging. What must come out the other side is the log the queue could
//! hold plus one line saying how much was lost — and the child must reach
//! its own exit rather than sit in `write`.
//!
//! A separate binary, and a separate process, because `init_tracing`
//! installs a process-global subscriber exactly once.

use std::io::{BufRead as _, Read as _, Write as _};
use std::time::{Duration, Instant};

const CHILD_ENV: &str = "AISIX_LOG_BACKPRESSURE_CHILD";
/// Far more than the queue holds, so the drop path is certainly taken.
const EVENTS: usize = 200_000;
/// Marker the child writes to stdout once every event has been emitted.
const EMITTED: &str = "child-emitted-after-ms=";

#[test]
fn a_log_consumer_that_stops_reading_costs_lines_not_progress() {
    if std::env::var_os(CHILD_ENV).is_some() {
        return child();
    }

    let mut child = std::process::Command::new(std::env::current_exe().expect("test binary"))
        .env(CHILD_ENV, "1")
        // The child is this same binary, so point libtest at this test.
        .args(["--exact", "a_log_consumer_that_stops_reading_costs_lines_not_progress"])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("spawn the child");

    // The child's stdout is a second pipe, and it is the only thing the
    // parent reads until the marker arrives: while stderr sits unread the
    // child must still reach the end of its 200,000 events. A subscriber
    // that writes straight to stderr cannot, and this wait is what fails.
    let (marker_tx, marker_rx) = std::sync::mpsc::channel();
    let stdout = child.stdout.take().expect("child stdout");
    // Line by line, because the child does not exit until the parent has
    // drained stderr, which happens after this — but keep draining to EOF
    // afterwards, or the child's own harness output hits a closed pipe.
    std::thread::spawn(move || {
        let mut marker = Some(marker_tx);
        for line in std::io::BufReader::new(stdout)
            .lines()
            .map_while(Result::ok)
        {
            if line.contains(EMITTED) {
                if let Some(tx) = marker.take() {
                    let _ = tx.send(line);
                }
            }
        }
    });
    let mut stderr = child.stderr.take().expect("child stderr");
    let out = marker_rx
        .recv_timeout(Duration::from_secs(30))
        .expect("the child never finished emitting — it blocked on the unread log sink");

    // Now drain the log the child has been queueing.
    let mut log = String::new();
    stderr
        .read_to_string(&mut log)
        .expect("read the child's log");
    let status = child.wait().expect("child exits");

    assert!(
        status.success(),
        "child failed: {status}, log tail: {log:?}"
    );
    assert!(out.contains(EMITTED), "child stdout: {out}");
    assert!(
        log.contains("benchmark event"),
        "the queue's worth of events must still reach the log",
    );
    assert!(
        log.contains("dropped log events") && log.contains("dropped="),
        "a drained sink must state how many events were lost; log tail: {}",
        log.lines().rev().take(5).collect::<Vec<_>>().join(" | "),
    );
}

fn child() {
    let cfg = aisix_core::ObservabilityConfig {
        log_level: "info".into(),
        ..Default::default()
    };
    aisix_obs::init_tracing(&cfg).expect("install the subscriber");
    let started = Instant::now();
    for n in 0..EVENTS {
        tracing::info!(n, "benchmark event");
    }
    let emitted = started.elapsed();
    // Straight to the descriptor: libtest captures the print macros.
    let mut stdout = std::io::stdout();
    let _ = writeln!(stdout, "{EMITTED}{}", emitted.as_millis());
    let _ = stdout.flush();
    // Long enough for the parent to start reading and the queue to drain.
    aisix_obs::shutdown_logging(Duration::from_secs(30));
}
