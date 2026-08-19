//! Request-owned trace identity for the OTLP export path (AISIX-Cloud#1279).
//!
//! Before this module every exported span minted fresh random trace/span
//! ids inside the encoder — which runs inside the sink pipeline's retry
//! loop, so a transient delivery failure re-encoded the batch and a
//! receiver that dedups on span id saw N distinct spans for one attempt.
//! And with every span its own root, the per-attempt events of one
//! request (#655) joined only on the `aisix.request_id` attribute, which
//! no trace waterfall understands.
//!
//! The fix is ownership: a [`RequestTraceBundle`] is minted ONCE per
//! request (by the proxy's request-id middleware, the same single mint
//! point the request id itself uses), carries the trace id, the SERVER
//! span id, the logical GenAI span id and one span id per upstream
//! attempt, and rides into the sink layer as an immutable
//! [`TraceEmission`] snapshot on each record — so a delivery retry
//! re-encodes byte-identical ids, and every exporter sees the same trace.
//!
//! Wire vocabulary follows W3C Trace Context Level 1
//! (<https://www.w3.org/TR/trace-context/>): `traceparent` version 00,
//! 16-byte trace id / 8-byte parent id, lowercase hex. Ingestion is
//! strict — anything malformed degrades to a locally-rooted trace and the
//! caller's `tracestate` is discarded with it, so an upstream proxy that
//! mangles the header can never smuggle a zero id or mixed-case hex into
//! the export.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// The W3C `traceparent` request header (Trace Context Level 1 §3.2).
pub const TRACEPARENT_HEADER: &str = "traceparent";
/// The W3C `tracestate` request header (Trace Context Level 1 §3.3).
pub const TRACESTATE_HEADER: &str = "tracestate";

/// Longest accepted `tracestate` value. The spec permits up to 32 list
/// members; 512 bytes covers every real-world vendor chain while keeping
/// a hostile caller from riding kilobytes into every exported span.
const MAX_TRACESTATE_BYTES: usize = 512;

/// A 16-byte W3C trace id. All-zero is invalid per spec and unrepresentable
/// through the public constructors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TraceId(pub [u8; 16]);

/// An 8-byte W3C span id. All-zero is invalid per spec and unrepresentable
/// through the public constructors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpanId(pub [u8; 8]);

impl TraceId {
    /// 16 random bytes. UUIDv4 gives 122 random bits, which is the same
    /// source the request-id mint uses; the 6 version/variant bits are an
    /// acceptable loss for an id whose only job is uniqueness.
    pub fn random() -> Self {
        Self(*uuid::Uuid::new_v4().as_bytes())
    }

    /// 32 lowercase-hex chars, the OTLP/JSON and W3C wire form.
    pub fn to_hex(self) -> String {
        hex_lower(&self.0)
    }
}

impl SpanId {
    pub fn random() -> Self {
        let b = uuid::Uuid::new_v4();
        let b = b.as_bytes();
        Self([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
    }

    /// 16 lowercase-hex chars, the OTLP/JSON and W3C wire form.
    pub fn to_hex(self) -> String {
        hex_lower(&self.0)
    }
}

fn hex_lower(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        // Infallible for String.
        let _ = write!(s, "{b:02x}");
    }
    s
}

/// A validated inbound W3C trace context — the caller's own tracing,
/// which the request's SERVER span parents under so the caller's backend
/// shows the gateway inside the caller's trace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoteTraceContext {
    pub trace_id: TraceId,
    pub parent_span_id: SpanId,
    /// The verbatim `trace-flags` octet. Carried onto the exported SERVER
    /// span; deliberately NOT consulted by the exporter's own sampling —
    /// the operator's `sample_rate` stays authoritative, so an inbound
    /// `sampled=1` cannot force export past `sample_rate=0`.
    pub flags: u8,
    /// The caller's `tracestate`, kept only when the `traceparent` it
    /// travelled with was valid (spec: state is meaningless without its
    /// parent) and it passed the charset/length screen.
    pub tracestate: Option<String>,
}

/// Strictly parse one `traceparent` header value (W3C Trace Context
/// Level 1 §3.2). Accepted shape is exactly the version-00 form:
///
/// ```text
/// 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01
/// ```
///
/// Anything else — wrong length, uppercase hex, an unknown or `ff`
/// version, an all-zero trace or parent id — yields `None` and the
/// request gets a locally-rooted trace. The spec permits leniently
/// parsing future versions; a trust boundary does not: a value this
/// gateway will attach its own telemetry under has to be one it fully
/// understands.
pub fn parse_traceparent(value: &str) -> Option<(TraceId, SpanId, u8)> {
    let b = value.as_bytes();
    if b.len() != 55 || b[2] != b'-' || b[35] != b'-' || b[52] != b'-' {
        return None;
    }
    // Version: exactly "00". This also rejects "ff" (forbidden by spec)
    // and any future version we have not reviewed.
    if &value[0..2] != "00" {
        return None;
    }
    let trace_id = hex_bytes::<16>(&value[3..35])?;
    let parent_id = hex_bytes::<8>(&value[36..52])?;
    let flags = hex_bytes::<1>(&value[53..55])?[0];
    if trace_id == [0u8; 16] || parent_id == [0u8; 8] {
        return None;
    }
    Some((TraceId(trace_id), SpanId(parent_id), flags))
}

/// Decode exactly `2 * N` lowercase-hex chars. Uppercase is rejected —
/// the W3C wire form is lowercase-only and a mixed-case value signals a
/// producer we should not trust to have composed the rest correctly.
fn hex_bytes<const N: usize>(s: &str) -> Option<[u8; N]> {
    let b = s.as_bytes();
    if b.len() != N * 2 {
        return None;
    }
    let mut out = [0u8; N];
    for (i, chunk) in b.chunks_exact(2).enumerate() {
        let hi = hex_nibble(chunk[0])?;
        let lo = hex_nibble(chunk[1])?;
        out[i] = (hi << 4) | lo;
    }
    Some(out)
}

fn hex_nibble(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        _ => None,
    }
}

/// Screen a combined `tracestate` value: printable ASCII (0x20..=0x7E),
/// non-empty, within [`MAX_TRACESTATE_BYTES`]. The list-member grammar is
/// deliberately not enforced beyond that — the value is forwarded opaque
/// into the exported span's `traceState`, never interpreted.
pub fn screen_tracestate(value: &str) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty()
        || trimmed.len() > MAX_TRACESTATE_BYTES
        || !trimmed.bytes().all(|b| (0x20..=0x7E).contains(&b))
    {
        return None;
    }
    Some(trimmed.to_owned())
}

/// Wall-clock nanosecond timestamps measured on a monotonic base.
///
/// The wall clock is read ONCE, at request start; every later timestamp is
/// that base plus a monotonic `Instant` offset. This is what makes child
/// spans provably bracketed by their parents — a wall-clock step (NTP)
/// mid-request cannot reorder them — and what replaces the old
/// `occurred_at - latency` reconstruction, whose whole-second stamp put
/// ±1s of error on every span's absolute placement.
#[derive(Debug, Clone, Copy)]
pub struct TraceClock {
    base_unix_nanos: u64,
    base: Instant,
}

impl TraceClock {
    pub fn start() -> Self {
        let base_unix_nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos().min(u64::MAX as u128) as u64)
            .unwrap_or(0);
        Self {
            base_unix_nanos,
            base: Instant::now(),
        }
    }

    /// Nanoseconds since the Unix epoch, now.
    pub fn now_unix_nanos(&self) -> u64 {
        self.base_unix_nanos
            .saturating_add(self.base.elapsed().as_nanos().min(u64::MAX as u128) as u64)
    }
}

/// One upstream attempt's span identity and real dispatch boundaries.
#[derive(Debug, Clone, Copy)]
struct AttemptSpan {
    span_id: SpanId,
    start_unix_nano: u64,
    /// Stamped when the attempt settles ([`RequestTraceBundle::end_attempt`]);
    /// `None` for an attempt still in flight at emission time.
    end_unix_nano: Option<u64>,
}

/// The request-owned trace identity: minted once at request start,
/// mutated only through the attempt chokepoints, snapshotted into an
/// immutable [`TraceEmission`] per exported usage event.
#[derive(Debug)]
pub struct RequestTraceBundle {
    trace_id: TraceId,
    server_span_id: SpanId,
    /// The logical GenAI operation span — one per request, covering every
    /// retry/failover attempt per the GenAI semantic conventions.
    logical_span_id: SpanId,
    remote: Option<RemoteTraceContext>,
    clock: TraceClock,
    server_start_unix_nano: u64,
    attempts: Mutex<Vec<AttemptSpan>>,
    /// Set by the first terminal emission. A second terminal call (a
    /// double-emit bug upstream of here) degrades to an attempt-only
    /// emission instead of shipping a duplicate SERVER span.
    terminal_emitted: AtomicBool,
}

impl RequestTraceBundle {
    /// Mint the request's trace identity. `remote` is the caller's
    /// validated `traceparent`/`tracestate`, if any: it decides the trace
    /// id (the caller's trace continues through the gateway) and the
    /// SERVER span's parent. Without it the request roots a fresh trace.
    pub fn new(remote: Option<RemoteTraceContext>) -> Self {
        let clock = TraceClock::start();
        Self {
            trace_id: remote
                .as_ref()
                .map(|r| r.trace_id)
                .unwrap_or_else(TraceId::random),
            server_span_id: SpanId::random(),
            logical_span_id: SpanId::random(),
            remote,
            server_start_unix_nano: clock.now_unix_nanos(),
            clock,
            attempts: Mutex::new(Vec::new()),
            terminal_emitted: AtomicBool::new(false),
        }
    }

    /// The trace id every span of this request shares, as 32 lowercase-hex
    /// chars — the public correlation key (`UsageEvent::trace_id`).
    pub fn trace_id_hex(&self) -> String {
        self.trace_id.to_hex()
    }

    /// Whether this request continues a caller-supplied trace (a valid
    /// inbound `traceparent`).
    pub fn has_remote_parent(&self) -> bool {
        self.remote.is_some()
    }

    /// Mark attempt `index` as dispatched now: mint its span id and stamp
    /// its start. Called from `RoutingTelemetry::begin_attempt`, the
    /// single place an attempt begins. Indexes arrive sequentially; a gap
    /// (a dispatch path that skipped `begin_attempt`) is backfilled at the
    /// same instant so the vec stays index-addressable.
    pub fn start_attempt(&self, index: u32) {
        let now = self.clock.now_unix_nanos();
        let mut attempts = self.attempts.lock().expect("trace bundle lock");
        while attempts.len() <= index as usize {
            attempts.push(AttemptSpan {
                span_id: SpanId::random(),
                start_unix_nano: now,
                end_unix_nano: None,
            });
        }
    }

    /// Stamp attempt `index`'s end, once. Called from
    /// `RoutingTelemetry::record`, the single place an attempt settles.
    /// For the winning streaming attempt this fires at commit time; the
    /// terminal emission later extends the end to the real stream end via
    /// the event's own measured duration (see [`Self::emission`]).
    pub fn end_attempt(&self, index: u32) {
        let now = self.clock.now_unix_nanos();
        let mut attempts = self.attempts.lock().expect("trace bundle lock");
        if let Some(a) = attempts.get_mut(index as usize) {
            a.end_unix_nano.get_or_insert(now);
        }
    }

    /// Snapshot the spans one usage event carries.
    ///
    /// Non-terminal (`terminal = false`): the event describes one failed /
    /// superseded attempt — emit that attempt's CLIENT span alone; the
    /// SERVER and logical spans ride the terminal event.
    ///
    /// Terminal (`terminal = true`, first such call): the request is done —
    /// stamp the SERVER span's end at NOW (the response body's real
    /// EOF/drop, since every terminal emit runs from the family's Drop
    /// guard or tail), emit SERVER + logical + this event's attempt span.
    /// The event's attempt end is extended to `start + upstream_latency`
    /// when the handler measured a longer duration than the commit-time
    /// stamp — the streaming case, where `record()` fires at commit but
    /// the upstream keeps streaming.
    ///
    /// A request that never dispatched an attempt (`attempts` empty):
    /// - `upstream_latency_ms > 0` — a family that dispatches without
    ///   per-attempt tracking (MCP / A2A / jobs / realtime): one CLIENT
    ///   span under SERVER, placed by the handler's own measured duration
    ///   ending now.
    /// - otherwise — a cache hit, a guardrail block, a pre-dispatch error:
    ///   the SERVER span alone. No fictitious upstream CLIENT span.
    pub fn emission(
        &self,
        terminal: bool,
        attempt_index: u32,
        upstream_latency_ms: u32,
    ) -> TraceEmission {
        let now = self.clock.now_unix_nanos();
        let latency_nanos = u64::from(upstream_latency_ms).saturating_mul(1_000_000);
        let attempts = self.attempts.lock().expect("trace bundle lock");
        let mut spans = Vec::new();

        // First terminal call wins; a duplicate degrades to attempt-only.
        // (`&&` short-circuits, so a non-terminal call never touches the
        // flag.)
        let terminal = terminal && !self.terminal_emitted.swap(true, Ordering::AcqRel);

        let attempt_span = attempts.get(attempt_index as usize).map(|a| {
            let mut end = a.end_unix_nano.unwrap_or(now);
            if terminal {
                // The winning streaming attempt: `record()` stamped the
                // commit, the handler measured the full stream. Both are
                // real monotonic measurements from the same dispatch
                // boundary; take the longer, clamped to now.
                end = end
                    .max(a.start_unix_nano.saturating_add(latency_nanos))
                    .min(now);
            }
            SpanEmit {
                role: SpanRole::Attempt,
                span_id: a.span_id,
                parent_span_id: Some(self.logical_span_id),
                start_unix_nano: a.start_unix_nano,
                end_unix_nano: end,
            }
        });
        // A non-terminal event from a dispatch path outside the attempt
        // chokepoints — an ensemble panel member or judge sub-call, the
        // known trunk bypass — still describes real upstream work:
        // synthesize its CLIENT span (placed by the handler's own measured
        // duration, ending now) so the sub-call is not lost from the
        // trace. The id is minted once per emission snapshot, so delivery
        // retries and every exporter still agree on it.
        let attempt_span = attempt_span.or_else(|| {
            (!terminal && upstream_latency_ms > 0).then(|| SpanEmit {
                role: SpanRole::Attempt,
                span_id: SpanId::random(),
                parent_span_id: Some(self.logical_span_id),
                start_unix_nano: now
                    .saturating_sub(latency_nanos)
                    .max(self.server_start_unix_nano),
                end_unix_nano: now,
            })
        });

        if terminal {
            spans.push(SpanEmit {
                role: SpanRole::Server,
                span_id: self.server_span_id,
                parent_span_id: self.remote.as_ref().map(|r| r.parent_span_id),
                start_unix_nano: self.server_start_unix_nano,
                end_unix_nano: now,
            });
            if let Some(first) = attempts.first() {
                // Logical span: first dispatch → last settle (extended to
                // this terminal event's own measured end).
                let last_end = attempts
                    .iter()
                    .filter_map(|a| a.end_unix_nano)
                    .max()
                    .unwrap_or(now)
                    .max(attempt_span.as_ref().map(|s| s.end_unix_nano).unwrap_or(0))
                    .min(now);
                spans.push(SpanEmit {
                    role: SpanRole::Logical,
                    span_id: self.logical_span_id,
                    parent_span_id: Some(self.server_span_id),
                    start_unix_nano: first.start_unix_nano,
                    end_unix_nano: last_end,
                });
            } else if upstream_latency_ms > 0 {
                // Dispatched upstream without per-attempt tracking: one
                // CLIENT span placed by the handler's measured duration,
                // clamped inside the SERVER span.
                spans.push(SpanEmit {
                    role: SpanRole::Logical,
                    span_id: self.logical_span_id,
                    parent_span_id: Some(self.server_span_id),
                    start_unix_nano: now
                        .saturating_sub(latency_nanos)
                        .max(self.server_start_unix_nano),
                    end_unix_nano: now,
                });
            }
            // else: cache hit / block / pre-dispatch error — SERVER only.
        }

        spans.extend(attempt_span);

        TraceEmission {
            trace_id: self.trace_id,
            remote_flags: self.remote.as_ref().map(|r| r.flags),
            tracestate: self.remote.as_ref().and_then(|r| r.tracestate.clone()),
            spans,
        }
    }
}

/// Which structural role a span plays in the exported hierarchy. Decides
/// OTLP `kind` and which attribute set the encoder attaches; deliberately
/// NOT exported as a span attribute — consumers distinguish spans by
/// `kind` and parent linkage, the vocabulary trace backends already have.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpanRole {
    /// The inbound HTTP request: one per request, OTLP `SPAN_KIND_SERVER`.
    Server,
    /// The logical GenAI operation covering every attempt, OTLP
    /// `SPAN_KIND_CLIENT`. Doubles as the single upstream-call span for
    /// families without per-attempt tracking.
    Logical,
    /// One upstream dispatch attempt, OTLP `SPAN_KIND_CLIENT`.
    Attempt,
}

/// One span's identity and boundaries, ready to encode.
#[derive(Debug, Clone, Copy)]
pub struct SpanEmit {
    pub role: SpanRole,
    pub span_id: SpanId,
    /// `None` only for a SERVER span with no valid inbound `traceparent`
    /// — the trace's local root.
    pub parent_span_id: Option<SpanId>,
    pub start_unix_nano: u64,
    pub end_unix_nano: u64,
}

/// The immutable per-event snapshot the sink layer carries
/// (`SinkRecord::trace`). Built once at emission time, so the encoder —
/// which runs inside the delivery retry loop — reproduces byte-identical
/// ids and timestamps on every retry and across every exporter.
#[derive(Debug, Clone)]
pub struct TraceEmission {
    pub trace_id: TraceId,
    /// The verbatim inbound `trace-flags` octet when the request continued
    /// a remote trace; `None` for a locally-rooted one.
    pub remote_flags: Option<u8>,
    /// The caller's screened `tracestate`, exported on the SERVER span.
    pub tracestate: Option<String>,
    pub spans: Vec<SpanEmit>,
}

impl TraceEmission {
    /// The event's carrier span — the one the encoder attaches the usage
    /// event's full attribute set to: the attempt span when present, else
    /// the logical span, else the SERVER span.
    pub fn carrier_role(&self) -> Option<SpanRole> {
        [SpanRole::Attempt, SpanRole::Logical, SpanRole::Server]
            .into_iter()
            .find(|role| self.spans.iter().any(|s| s.role == *role))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const VALID: &str = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";

    #[test]
    fn parses_a_valid_v00_traceparent() {
        let (trace, parent, flags) = parse_traceparent(VALID).expect("valid header");
        assert_eq!(trace.to_hex(), "4bf92f3577b34da6a3ce929d0e0e4736");
        assert_eq!(parent.to_hex(), "00f067aa0ba902b7");
        assert_eq!(flags, 0x01);
    }

    #[test]
    fn rejects_malformed_traceparents() {
        for (case, why) in [
            ("", "empty"),
            (
                "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7",
                "missing flags",
            ),
            (
                "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01-extra",
                "trailing segment",
            ),
            (
                "ff-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
                "forbidden version ff",
            ),
            (
                "01-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
                "unreviewed future version",
            ),
            (
                "00-00000000000000000000000000000000-00f067aa0ba902b7-01",
                "all-zero trace id",
            ),
            (
                "00-4bf92f3577b34da6a3ce929d0e0e4736-0000000000000000-01",
                "all-zero parent id",
            ),
            (
                "00-4BF92F3577B34DA6A3CE929D0E0E4736-00f067aa0ba902b7-01",
                "uppercase hex",
            ),
            (
                "00-4bf92f3577b34da6a3ce929d0e0e473g-00f067aa0ba902b7-01",
                "non-hex char",
            ),
            (
                "00_4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
                "wrong delimiter",
            ),
            (
                "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-1",
                "one-char flags",
            ),
        ] {
            assert!(parse_traceparent(case).is_none(), "must reject: {why}");
        }
    }

    #[test]
    fn tracestate_screen_keeps_sane_values_and_drops_hostile_ones() {
        assert_eq!(
            screen_tracestate("congo=t61rcWkgMzE,rojo=00f067aa0ba902b7").as_deref(),
            Some("congo=t61rcWkgMzE,rojo=00f067aa0ba902b7")
        );
        assert_eq!(
            screen_tracestate("  vendor=x  ").as_deref(),
            Some("vendor=x")
        );
        assert!(screen_tracestate("").is_none());
        assert!(screen_tracestate("   ").is_none());
        assert!(screen_tracestate("bad\nvalue").is_none(), "control chars");
        assert!(screen_tracestate("non-ascii-日本").is_none());
        assert!(screen_tracestate(&"x".repeat(MAX_TRACESTATE_BYTES + 1)).is_none());
    }

    #[test]
    fn clock_is_monotonic_and_epoch_anchored() {
        let clock = TraceClock::start();
        let a = clock.now_unix_nanos();
        let b = clock.now_unix_nanos();
        assert!(b >= a);
        // Sanity: after 2020-01-01 in nanos.
        assert!(a > 1_577_836_800_000_000_000);
    }

    fn bundle_with_remote() -> RequestTraceBundle {
        let (trace_id, parent_span_id, flags) = parse_traceparent(VALID).unwrap();
        RequestTraceBundle::new(Some(RemoteTraceContext {
            trace_id,
            parent_span_id,
            flags,
            tracestate: Some("vendor=x".into()),
        }))
    }

    #[test]
    fn remote_parent_decides_trace_id_and_server_parent() {
        let bundle = bundle_with_remote();
        assert_eq!(bundle.trace_id_hex(), "4bf92f3577b34da6a3ce929d0e0e4736");
        let em = bundle.emission(true, 0, 0);
        let server = em
            .spans
            .iter()
            .find(|s| s.role == SpanRole::Server)
            .expect("terminal emission carries the SERVER span");
        assert_eq!(
            server.parent_span_id.map(SpanId::to_hex).as_deref(),
            Some("00f067aa0ba902b7")
        );
        assert_eq!(em.remote_flags, Some(0x01));
        assert_eq!(em.tracestate.as_deref(), Some("vendor=x"));
    }

    #[test]
    fn local_root_has_no_parent_and_a_random_trace() {
        let bundle = RequestTraceBundle::new(None);
        let em = bundle.emission(true, 0, 0);
        let server = em
            .spans
            .iter()
            .find(|s| s.role == SpanRole::Server)
            .unwrap();
        assert!(server.parent_span_id.is_none());
        assert!(em.remote_flags.is_none());
        assert_ne!(bundle.trace_id_hex(), "00000000000000000000000000000000");
    }

    /// The failover shape the whole design exists for: SERVER → logical →
    /// N attempt children, ids stable across snapshots, children bracketed
    /// by parents.
    #[test]
    fn failover_emissions_share_one_hierarchy() {
        let bundle = RequestTraceBundle::new(None);
        bundle.start_attempt(0);
        bundle.end_attempt(0);
        bundle.start_attempt(1);
        bundle.end_attempt(1);

        // The failed attempt's event (non-terminal).
        let failed = bundle.emission(false, 0, 40);
        assert_eq!(failed.spans.len(), 1, "attempt span only");
        let failed_span = failed.spans[0];
        assert_eq!(failed_span.role, SpanRole::Attempt);

        // The winner's terminal event.
        let terminal = bundle.emission(true, 1, 60);
        let server = terminal
            .spans
            .iter()
            .find(|s| s.role == SpanRole::Server)
            .unwrap();
        let logical = terminal
            .spans
            .iter()
            .find(|s| s.role == SpanRole::Logical)
            .unwrap();
        let attempt = terminal
            .spans
            .iter()
            .find(|s| s.role == SpanRole::Attempt)
            .unwrap();

        assert_eq!(failed.trace_id, terminal.trace_id);
        assert_eq!(logical.parent_span_id, Some(server.span_id));
        assert_eq!(attempt.parent_span_id, Some(logical.span_id));
        assert_eq!(failed_span.parent_span_id, Some(logical.span_id));
        assert_ne!(failed_span.span_id, attempt.span_id);

        // Bracketing: logical inside server, attempts inside logical.
        assert!(server.start_unix_nano <= logical.start_unix_nano);
        assert!(logical.end_unix_nano <= server.end_unix_nano);
        assert!(logical.start_unix_nano <= failed_span.start_unix_nano);
        assert!(attempt.end_unix_nano <= logical.end_unix_nano);
    }

    /// Ids must be identical across repeated snapshots — the delivery
    /// retry re-encodes from the same emission, but a second EXPORTER also
    /// snapshots independently, and both must agree.
    #[test]
    fn snapshots_are_id_stable() {
        let bundle = RequestTraceBundle::new(None);
        bundle.start_attempt(0);
        bundle.end_attempt(0);
        let a = bundle.emission(false, 0, 10);
        let b = bundle.emission(false, 0, 10);
        assert_eq!(a.trace_id, b.trace_id);
        assert_eq!(a.spans[0].span_id.to_hex(), b.spans[0].span_id.to_hex());
        assert_eq!(a.spans[0].start_unix_nano, b.spans[0].start_unix_nano);
    }

    /// The streaming winner: `record()` stamped commit-time, the handler
    /// measured the full stream — the longer real measurement wins, but
    /// never past the emission instant (a claimed duration cannot push a
    /// span end into the future).
    #[test]
    fn terminal_emission_extends_the_winning_attempt_to_the_measured_duration() {
        let bundle = RequestTraceBundle::new(None);
        bundle.start_attempt(0);
        bundle.end_attempt(0); // commit-time stamp, ~0ms after start
        std::thread::sleep(std::time::Duration::from_millis(15));
        // Terminal emission happens at real stream end; the handler
        // measured 10ms of streaming after the commit stamp.
        let em = bundle.emission(true, 0, 10);
        let attempt = em
            .spans
            .iter()
            .find(|s| s.role == SpanRole::Attempt)
            .unwrap();
        let dur = attempt.end_unix_nano - attempt.start_unix_nano;
        assert!(
            dur >= 10_000_000,
            "attempt span must cover the measured 10ms stream, got {dur}ns"
        );
        let server = em
            .spans
            .iter()
            .find(|s| s.role == SpanRole::Server)
            .unwrap();
        assert!(
            attempt.end_unix_nano <= server.end_unix_nano,
            "a measured duration must never push a span past the emission instant"
        );
    }

    /// Cache hits and pre-dispatch blocks: SERVER span alone — no
    /// fictitious upstream CLIENT span.
    #[test]
    fn no_attempts_and_no_latency_emits_server_only() {
        let bundle = RequestTraceBundle::new(None);
        let em = bundle.emission(true, 0, 0);
        assert_eq!(em.spans.len(), 1);
        assert_eq!(em.spans[0].role, SpanRole::Server);
        assert_eq!(em.carrier_role(), Some(SpanRole::Server));
    }

    /// Families that dispatch without per-attempt tracking (MCP / A2A /
    /// jobs): one CLIENT span under SERVER, placed by measured duration.
    #[test]
    fn no_attempts_with_latency_emits_one_client_child() {
        let bundle = RequestTraceBundle::new(None);
        let em = bundle.emission(true, 0, 25);
        let server = em
            .spans
            .iter()
            .find(|s| s.role == SpanRole::Server)
            .unwrap();
        let logical = em
            .spans
            .iter()
            .find(|s| s.role == SpanRole::Logical)
            .unwrap();
        assert_eq!(em.spans.len(), 2);
        assert_eq!(logical.parent_span_id, Some(server.span_id));
        assert!(logical.start_unix_nano >= server.start_unix_nano);
        assert!(logical.end_unix_nano <= server.end_unix_nano);
        assert_eq!(em.carrier_role(), Some(SpanRole::Logical));
    }

    /// A double terminal emit (an upstream bug) must not ship two SERVER
    /// spans.
    #[test]
    fn second_terminal_emission_degrades_to_attempt_only() {
        let bundle = RequestTraceBundle::new(None);
        bundle.start_attempt(0);
        bundle.end_attempt(0);
        let first = bundle.emission(true, 0, 10);
        assert!(first.spans.iter().any(|s| s.role == SpanRole::Server));
        let second = bundle.emission(true, 0, 10);
        assert!(
            !second.spans.iter().any(|s| s.role == SpanRole::Server),
            "duplicate SERVER span"
        );
    }
}
