//! Structured one-line access log — one line per request, success or error.
//! Keeping the call explicit rather than inside a tower layer means the
//! caller can attach `provider`, `model`, `api_key_id`, and `tokens` —
//! fields the layer couldn't see.
//!
//! # When the line is written, and what that costs
//!
//! Exactly one line per request, whatever the outcome — but WHEN its fields
//! are settled differs by path, and that decides which fields can be filled
//! at all. Four cases:
//!
//! - **Non-streamed response** — built by the handler, on its way out, with
//!   everything it resolved available.
//! - **Streamed response** — NOT when the SSE head is handed to the server.
//!   The handler defers the line to the request's attribution cell
//!   (`attribution::defer_access_log`) and it is completed beside the
//!   request's TERMINAL usage event, at the point the stream's outcome is
//!   known: fully consumed, abandoned mid-stream, or dropped before its
//!   first poll. It therefore reports the same `status`, `error_kind` and
//!   `error` as that event — a stream whose consumer walked away reads
//!   `499` / `client_disconnected` on both — and it can carry the token
//!   counts and `provider_request_id`, which only exist once the upstream
//!   has answered (AISIX-Cloud#1571).
//!
//!   The two relays of an open-ended binary body — `/v1/audio/speech` and
//!   `/v1/videos/{id}/content` — are completed at the relay's end the same
//!   way, so their `duration_ms` spans the relay like every other surface.
//! - **`/v1/realtime`** — the opposite extreme. The handler returns the
//!   WebSocket upgrade immediately; the line is written by `run_session` on
//!   a detached task once the session closes, so it carries the close status
//!   and the session's real token totals.
//! - **Caller hung up before the response head was written** — from
//!   `ClientCancelGuard::drop`, with no handler involved. Status is `499`,
//!   and the fields it can fill are the ones the request published to its
//!   attribution cell as it resolved: `model`, `provider`, and the
//!   dispatched target (`upstream_model` + `provider_key_id`). The
//!   handler-side figures — tokens, `provider_request_id`, the routing
//!   counts — stay `None`, because the future was dropped before it could
//!   produce them. Such a request also emits a `499` usage event carrying
//!   the same identities, keyed by this `request_id`.
//!
//! So do not add a field whose value only exists once the upstream has
//! responded and expect it on every line: it is silently empty on the
//! cancelled ones, where the request never got that far.
//!
//! Whichever of those builds the line, it is WRITTEN once the response body
//! is finished with — fully handed to the HTTP server, or dropped because
//! the client went away — because that is the first moment the two body
//! sizes below are known. `/v1/realtime` and the head-phase cancel have no
//! response body to wait for and write theirs at once.
//!
//! # Body sizes
//!
//! - `request_body_bytes` is how many request-body bytes the gateway read
//!   from the client. The gateway never decompresses a request body, so
//!   this is both the size on the wire and the size it parsed. It is
//!   absent unless the body was read to its end: a request refused before
//!   its body was read (the `Content-Length` pre-check's `413`), a route
//!   that never reads one (most `GET`s), a client that hung up before or
//!   during the upload, and `/v1/realtime`, whose WebSocket has no HTTP
//!   body.
//! - `response_body_bytes` is every response-body byte handed to the HTTP
//!   server for the client — SSE framing and keep-alive heartbeats
//!   included, headers, chunked-transfer framing and TLS excluded. A
//!   client that disconnects mid-response gets what was handed over until
//!   then. It is absent only when no response head was written — the
//!   head-phase cancel line — and on `/v1/realtime`.
//!
//! A fifth case is not about WHEN the line is written but about what
//! happened: a **cache hit** is written from the handler like any other
//! buffered response, and contacted no upstream at all. Its line says so
//! through [`CacheAccessLog`], and the fields that describe a dispatch
//! (`upstream_model`, `provider_key_id`, `served_by_model`,
//! `provider_request_id`) report only what the request can honestly claim
//! without one — see `upstream_model` below.
//!
//! # `latency` and `duration` answer two different questions
//!
//! - `latency_ms` is what the CALLER waited for: the first token forwarded
//!   downstream on a streamed response, the complete response on a buffered
//!   one. It is the same figure the request's terminal `UsageEvent` reports
//!   as `downstream_latency_ms`. Deliberately not the length of the stream
//!   (AISIX-Cloud#1394) — reading a minutes-long stream's wait as its
//!   time-to-first-token is what makes a working stream look like a
//!   connection sitting idle.
//! - `duration_ms` is how long the request occupied the gateway, arrival to
//!   last byte out. On a non-streamed request the two coincide; on a
//!   streamed one they differ by the whole length of the stream.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

/// `observability.access_log`, installed once by [`crate::init_tracing`].
/// Checked where every access-log line is written rather than expressed
/// as a filter directive, so no `RUST_LOG` spelling can bring the lines
/// back and no other event is affected.
static ENABLED: AtomicBool = AtomicBool::new(true);

pub(crate) fn set_enabled(enabled: bool) {
    ENABLED.store(enabled, Ordering::Relaxed);
}

/// Canonical access-log fields, passed to [`log_access`].
///
/// Constructed at the point a request's outcome becomes known — which is not
/// the same moment, nor even the same caller, on every path. See the module
/// docs before assuming a field is available here.
#[derive(Debug, Clone)]
pub struct AccessLog<'a> {
    pub method: &'a str,
    pub path: &'a str,
    pub status: u16,
    /// What the caller waited for — see the module docs. On a streamed
    /// response this is time-to-first-token, NOT how long the stream ran.
    pub latency: Duration,
    /// How long the request occupied the gateway: from arrival to the point
    /// the request's outcome was settled — the end of the stream on a
    /// streamed response, the end of the relay on `/v1/audio/speech` and
    /// `/v1/videos/{id}/content`. Equal to `latency` on every other
    /// non-streamed response.
    pub duration: Duration,
    /// Vendor id of the target that served the request. Unlike the pair
    /// below it reports the `unknown` SENTINEL rather than being omitted
    /// when nothing resolved one — a cache hit on a Model Group, an
    /// ensemble — because it is the same string the Prometheus `provider`
    /// label carries for that request, where a label cannot be absent. The
    /// two are read together often enough that spelling one condition two
    /// ways costs more than the sentinel does; `cache_status` is what says
    /// whether the cache is the reason.
    pub provider: Option<&'a str>,
    /// The model name the CALLER addressed — for a routing group, the group
    /// itself, never the target it dispatched to. See `upstream_model`
    /// below for the other half.
    pub model: Option<&'a str>,
    /// The upstream model id of the target this request last selected, and
    /// the ProviderKey it dispatched through. `model` alone cannot answer
    /// "which provider actually served this", and on a line written before
    /// any response exists — a `499` — nothing else names the target at all
    /// (AISIX-Cloud#1571). Both are `None` until a target was selected, and
    /// on the emitters that run detached from the request task.
    ///
    /// **A cache hit (`cache.status == "hit"`) selected no target**, and
    /// these two do not claim one. What a hit may still carry is the
    /// entry's own static mapping: a direct model's `model_name` and
    /// `provider_key_id` are properties of the row the caller addressed,
    /// true whether or not a request ever left the gateway. A Model Group
    /// has neither of its own, and nothing records which of its targets
    /// produced the stored entry, so a group's hit line carries neither
    /// field rather than naming a target that did not run.
    pub upstream_model: Option<&'a str>,
    pub provider_key_id: Option<&'a str>,
    pub api_key_id: Option<&'a str>,
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
    pub request_id: &'a str,
    /// Provider response object `id` of the attempt that served the request
    /// — OpenAI's `chat.completion.id`, Anthropic's message `id`,
    /// `/v1/responses`' `resp_…`. Distinct from `request_id` (this
    /// gateway's own id) and from the provider's HTTP transport header id;
    /// none of the three may overwrite another (AISIX-Cloud#1289).
    ///
    /// `None` whenever no id exists by the time this line is written:
    /// the request never reached an upstream (guardrail block,
    /// pre-dispatch error), it was served from cache, or the endpoint's
    /// provider response carries no id at all (embeddings / audio /
    /// images / count_tokens). A **streamed** response does carry it —
    /// the id arrives in the first frame and the line is written at the
    /// stream's end (AISIX-Cloud#1571) — unless the caller walked away
    /// before that frame. Mid-stream-failed-over calls are covered by the
    /// per-attempt `provider call completed` line (see
    /// `UsageSink::try_emit`), which shares this `request_id`.
    pub provider_request_id: Option<&'a str>,
    /// Routing target that ultimately served the request (the winning
    /// attempt's display name). `None` for direct models / cache hits.
    pub served_by_model: Option<&'a str>,
    /// Total upstream attempts made (initial + retries + fallbacks).
    pub routing_attempt_count: Option<u32>,
    /// How many attempts moved to a different target. Per #655 the
    /// per-attempt detail lives in telemetry (per-attempt UsageEvents),
    /// not in this one-line-per-request access log.
    pub routing_fallback_count: Option<u32>,
    /// Stable failure class (`ProxyError::kind`) — `None` on success.
    /// Machine-readable so an operator can filter or alert on a class
    /// without parsing the free-text message below.
    pub error_kind: Option<&'a str>,
    /// Why the request failed — `None` on success. Without it a 5xx line
    /// carries only `status` + `latency_ms`, which is the same shape for a
    /// kernel-level connect timeout, an upstream 500, and a blocked
    /// guardrail (AISIX-Cloud#1093).
    pub error: Option<&'a str>,
    /// What the request was, on the `/mcp` endpoints — `None` everywhere
    /// else. MCP tunnels every operation through one `POST`, so `method` and
    /// `path` alone describe nothing (#1181).
    pub mcp: Option<McpAccessLog<'a>>,
    /// How the response cache answered — `None` on every line that had no
    /// cache decision to report. See [`CacheAccessLog`].
    pub cache: Option<CacheAccessLog<'a>>,
    /// Request-body bytes the gateway read — `None` unless the body was
    /// read to its end. See the module docs.
    pub request_body_bytes: Option<u64>,
    /// Response-body bytes handed to the HTTP server — `None` when no
    /// response head was written. See the module docs.
    pub response_body_bytes: Option<u64>,
}

/// The response-cache half of an access-log line (AISIX-Cloud#1571).
///
/// Only `/v1/chat/completions` caches responses, and only its buffered
/// exit reports one: a streamed response is never cached, and a line
/// written before the handler produced a response — an error, a `499` —
/// had no cache decision to report at all.
///
/// Without it the line has NO marker for a cache hit, so a request served
/// entirely out of Redis is indistinguishable from one that went to the
/// provider except by the absence of fields that are also absent for other
/// reasons. The two names match the usage event's `cache_status` /
/// `cache_hit_layer` exactly, so a line and the row cp-api stores for the
/// same `request_id` read the same way.
#[derive(Debug, Clone, Default)]
pub struct CacheAccessLog<'a> {
    /// `disabled` / `miss` / `hit` / `bypass`, as the usage event spells
    /// it.
    pub status: &'a str,
    /// Which matching layer served a hit — `exact` or `semantic`. `None`
    /// on every non-hit status.
    pub hit_layer: Option<&'a str>,
}

/// Why a request addressed to a semantic router was not served by a
/// matched route. Rendered as `semantic_fallback`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SemanticFallback {
    /// Embedding decision: no route reached its threshold.
    NoMatch,
    /// Classifier decision: the model picked a route with a confidence
    /// below `min_confidence`.
    LowConfidence,
    /// The request carried no user text to decide on.
    EmptyPrompt,
    /// The embedding or decision call failed or timed out, and the
    /// router's failure policy applied.
    DecisionFailed,
    /// A route matched, but its target could not serve this caller (its
    /// client-IP allowlist excluded the caller, or it was unavailable while
    /// `default` was not), so `default` served instead.
    TargetUnavailable,
}

impl SemanticFallback {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NoMatch => "no_match",
            Self::LowConfidence => "low_confidence",
            Self::EmptyPrompt => "empty_prompt",
            Self::DecisionFailed => "decision_failed",
            Self::TargetUnavailable => "target_unavailable",
        }
    }
}

/// The semantic-router half of an access-log line: how the router the
/// caller addressed decided. Present only on requests addressed to a
/// semantic router, whichever way the request ended.
///
/// Owned rather than borrowed like the other halves, because it is not
/// built with the line: the router decides long before the line exists,
/// and publishes the decision where every writer of the request's line can
/// read it.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SemanticAccessLog {
    /// `semantic_route`: the route that served the request. `None` when
    /// `default` or a failure target served it.
    pub route: Option<String>,
    /// `semantic_score`: the deciding route's cosine similarity (embedding
    /// routers) or the decision model's confidence in its pick (classifier
    /// routers). Present whenever a decision was computed, including one
    /// that fell back because the score was too low; `None` when no
    /// decision was computed.
    pub score: Option<f64>,
    /// `semantic_fallback`: set only when no matched route served the
    /// request.
    pub fallback: Option<SemanticFallback>,
}

/// The `/mcp` half of an access-log line: which JSON-RPC method the single
/// `POST` carried, and enough of its outcome to tell the three ways a
/// `tools/list` can come back empty apart.
#[derive(Debug, Clone, Default)]
pub struct McpAccessLog<'a> {
    /// JSON-RPC `method` — `initialize`, `tools/list`, `tools/call`, …
    /// `None` when the body is not a single JSON-RPC message.
    pub method: Option<&'a str>,
    /// `tools/call` only: the tool name as the caller sent it.
    pub tool: Option<&'a str>,
    /// `tools/list` only: tools the upstreams returned, summed, before the
    /// caller's ACL filtered them.
    pub tools_total: Option<u32>,
    /// `tools/list` only: tools left after ACL filtering — what the caller
    /// actually received.
    pub tools_returned: Option<u32>,
}

impl AccessLog<'_> {
    /// Emit a single `tracing::info!` event carrying every field, unless
    /// `observability.access_log` is off. When on, the line is filtered by
    /// the log level like any other `info` event.
    pub fn emit(&self) {
        self.emit_with(None);
    }

    fn emit_with(&self, semantic: Option<&SemanticAccessLog>) {
        if !ENABLED.load(Ordering::Relaxed) {
            return;
        }
        let mcp = self.mcp.as_ref();
        tracing::info!(
            method = self.method,
            path = self.path,
            status = self.status,
            latency_ms = self.latency.as_millis() as u64,
            duration_ms = self.duration.as_millis() as u64,
            provider = self.provider,
            model = self.model,
            upstream_model = self.upstream_model,
            provider_key_id = self.provider_key_id,
            api_key_id = self.api_key_id,
            prompt_tokens = self.prompt_tokens,
            completion_tokens = self.completion_tokens,
            total_tokens = self.total_tokens,
            request_id = self.request_id,
            provider_request_id = self.provider_request_id,
            served_by_model = self.served_by_model,
            routing_attempt_count = self.routing_attempt_count,
            routing_fallback_count = self.routing_fallback_count,
            error_kind = self.error_kind,
            error = self.error,
            cache_status = self.cache.as_ref().map(|c| c.status),
            cache_hit_layer = self.cache.as_ref().and_then(|c| c.hit_layer),
            mcp_method = mcp.and_then(|m| m.method),
            mcp_tool = mcp.and_then(|m| m.tool),
            tools_total = mcp.and_then(|m| m.tools_total),
            tools_returned = mcp.and_then(|m| m.tools_returned),
            request_body_bytes = self.request_body_bytes,
            response_body_bytes = self.response_body_bytes,
            semantic_route = semantic.and_then(|s| s.route.as_deref()),
            semantic_score = semantic.and_then(|s| s.score),
            semantic_fallback = semantic.and_then(|s| s.fallback.map(SemanticFallback::as_str)),
            "proxy request completed",
        );
    }

    /// An owned copy, for a line whose body sizes are only known after the
    /// code that built it has returned.
    pub fn to_record(&self) -> AccessLogRecord {
        let owned = |v: Option<&str>| v.map(str::to_owned);
        AccessLogRecord {
            method: self.method.to_owned(),
            path: self.path.to_owned(),
            status: self.status,
            latency: self.latency,
            duration: self.duration,
            provider: owned(self.provider),
            model: owned(self.model),
            upstream_model: owned(self.upstream_model),
            provider_key_id: owned(self.provider_key_id),
            api_key_id: owned(self.api_key_id),
            prompt_tokens: self.prompt_tokens,
            completion_tokens: self.completion_tokens,
            total_tokens: self.total_tokens,
            request_id: self.request_id.to_owned(),
            provider_request_id: owned(self.provider_request_id),
            served_by_model: owned(self.served_by_model),
            routing_attempt_count: self.routing_attempt_count,
            routing_fallback_count: self.routing_fallback_count,
            error_kind: owned(self.error_kind),
            error: owned(self.error),
            mcp: self.mcp.as_ref().map(|m| {
                (
                    owned(m.method),
                    owned(m.tool),
                    m.tools_total,
                    m.tools_returned,
                )
            }),
            cache: self
                .cache
                .as_ref()
                .map(|c| (c.status.to_owned(), owned(c.hit_layer))),
            request_body_bytes: self.request_body_bytes,
            response_body_bytes: self.response_body_bytes,
            semantic: None,
        }
    }
}

/// An owned [`AccessLog`]. The fields the writer completes after the fact
/// are public; the rest are fixed when the copy is taken.
#[derive(Debug, Clone)]
pub struct AccessLogRecord {
    method: String,
    path: String,
    status: u16,
    latency: Duration,
    pub duration: Duration,
    provider: Option<String>,
    model: Option<String>,
    upstream_model: Option<String>,
    provider_key_id: Option<String>,
    api_key_id: Option<String>,
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
    total_tokens: Option<u64>,
    request_id: String,
    provider_request_id: Option<String>,
    served_by_model: Option<String>,
    routing_attempt_count: Option<u32>,
    routing_fallback_count: Option<u32>,
    error_kind: Option<String>,
    error: Option<String>,
    #[allow(clippy::type_complexity)]
    mcp: Option<(Option<String>, Option<String>, Option<u32>, Option<u32>)>,
    cache: Option<(String, Option<String>)>,
    pub request_body_bytes: Option<u64>,
    pub response_body_bytes: Option<u64>,
    /// Completed by the writer from what the request published, never by
    /// the code that built the line — see [`SemanticAccessLog`].
    pub semantic: Option<SemanticAccessLog>,
}

impl AccessLogRecord {
    pub fn emit(&self) {
        fn borrowed(v: &Option<String>) -> Option<&str> {
            v.as_deref()
        }
        AccessLog {
            method: &self.method,
            path: &self.path,
            status: self.status,
            latency: self.latency,
            duration: self.duration,
            provider: borrowed(&self.provider),
            model: borrowed(&self.model),
            upstream_model: borrowed(&self.upstream_model),
            provider_key_id: borrowed(&self.provider_key_id),
            api_key_id: borrowed(&self.api_key_id),
            prompt_tokens: self.prompt_tokens,
            completion_tokens: self.completion_tokens,
            total_tokens: self.total_tokens,
            request_id: &self.request_id,
            provider_request_id: borrowed(&self.provider_request_id),
            served_by_model: borrowed(&self.served_by_model),
            routing_attempt_count: self.routing_attempt_count,
            routing_fallback_count: self.routing_fallback_count,
            error_kind: borrowed(&self.error_kind),
            error: borrowed(&self.error),
            mcp: self
                .mcp
                .as_ref()
                .map(|(method, tool, total, returned)| McpAccessLog {
                    method: method.as_deref(),
                    tool: tool.as_deref(),
                    tools_total: *total,
                    tools_returned: *returned,
                }),
            cache: self.cache.as_ref().map(|(status, layer)| CacheAccessLog {
                status,
                hit_layer: layer.as_deref(),
            }),
            request_body_bytes: self.request_body_bytes,
            response_body_bytes: self.response_body_bytes,
        }
        .emit_with(self.semantic.as_ref());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tracing::subscriber::with_default;
    use tracing_subscriber::fmt::MakeWriter;
    use tracing_subscriber::{fmt, EnvFilter};

    /// Collect emitted log bytes into an in-memory buffer.
    #[derive(Clone, Default)]
    struct VecWriter {
        buf: std::sync::Arc<std::sync::Mutex<Vec<u8>>>,
    }
    impl VecWriter {
        fn contents(&self) -> String {
            String::from_utf8_lossy(&self.buf.lock().unwrap()).into_owned()
        }
    }
    impl std::io::Write for VecWriter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.buf.lock().unwrap().extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    impl<'a> MakeWriter<'a> for VecWriter {
        type Writer = VecWriter;
        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    #[test]
    fn emit_writes_every_field_into_the_subscriber() {
        let writer = VecWriter::default();
        let subscriber = fmt()
            .with_writer(writer.clone())
            .with_ansi(false)
            .with_target(false)
            .with_env_filter(EnvFilter::new("info"))
            .finish();

        with_default(subscriber, || {
            AccessLog {
                method: "POST",
                path: "/v1/chat/completions",
                status: 200,
                latency: Duration::from_millis(42),
                duration: Duration::from_millis(9_000),
                provider: Some("openai"),
                model: Some("my-gpt4"),
                upstream_model: Some("gpt-4o"),
                provider_key_id: Some("pk-1"),
                api_key_id: Some("key-id-1"),
                prompt_tokens: Some(2),
                completion_tokens: Some(1),
                total_tokens: Some(3),
                request_id: "req-abc",
                provider_request_id: Some("chatcmpl-abc"),
                served_by_model: Some("fallback-target"),
                routing_attempt_count: Some(2),
                routing_fallback_count: Some(1),
                error_kind: None,
                error: None,
                mcp: None,
                cache: None,
                request_body_bytes: Some(123),
                response_body_bytes: Some(4567),
            }
            .to_record()
            .emit();
        });

        let out = writer.contents();
        assert!(out.contains("proxy request completed"));
        // Through the owned copy too, which is how a deferred line is
        // written: the two sizes must survive it, and not be swapped.
        assert!(out.contains("request_body_bytes=123"), "{out}");
        assert!(out.contains("response_body_bytes=4567"), "{out}");
        assert!(out.contains("method=\"POST\"") || out.contains("method=POST"));
        assert!(out.contains("status=200"));
        assert!(out.contains("latency_ms=42"));
        // AISIX-Cloud#1571: the two figures are separate fields because on
        // a streamed line they are separate questions — what the caller
        // waited for, and how long the request held the gateway. This one
        // is deliberately the longer of the two, so transposing them at the
        // emit site cannot pass.
        assert!(out.contains("duration_ms=9000"), "{out}");
        assert!(out.contains("provider=\"openai\"") || out.contains("provider=openai"));
        assert!(out.contains("total_tokens=3"));
        assert!(out.contains("request_id=\"req-abc\"") || out.contains("request_id=req-abc"));
        // AISIX-Cloud#1289: the provider's own response id, next to — never
        // instead of — the gateway's `request_id`.
        assert!(
            out.contains("provider_request_id=\"chatcmpl-abc\"")
                || out.contains("provider_request_id=chatcmpl-abc"),
            "{out}"
        );
        assert!(
            out.contains("served_by_model=\"fallback-target\"")
                || out.contains("served_by_model=fallback-target")
        );
        assert!(out.contains("routing_attempt_count=2"));
        assert!(out.contains("routing_fallback_count=1"));
        // AISIX-Cloud#1571: `model` is what the caller addressed, so the
        // target it actually dispatched to has to be named separately —
        // otherwise a routing request's line says only the group, and a
        // `499` line names no target at all.
        assert!(
            out.contains("upstream_model=\"gpt-4o\"") || out.contains("upstream_model=gpt-4o"),
            "{out}"
        );
        assert!(
            out.contains("provider_key_id=\"pk-1\"") || out.contains("provider_key_id=pk-1"),
            "{out}"
        );
        // A success line must not carry failure fields at all — an
        // always-present `error=""` would defeat filtering on it.
        assert!(!out.contains("error_kind"), "{out}");
        assert!(!out.contains("error="), "{out}");
    }

    /// The gap this field closes: without it a failed request's only trace
    /// is `status=502 latency_ms=…`, identical for every cause.
    #[test]
    fn emit_carries_the_failure_class_and_reason() {
        let writer = VecWriter::default();
        let subscriber = fmt()
            .with_writer(writer.clone())
            .with_ansi(false)
            .with_target(false)
            .with_env_filter(EnvFilter::new("info"))
            .finish();

        with_default(subscriber, || {
            AccessLog {
                method: "POST",
                path: "/v1/messages",
                status: 504,
                latency: Duration::from_millis(7167),
                duration: Duration::from_millis(7167),
                provider: None,
                model: Some("claude-sonnet-4"),
                upstream_model: None,
                provider_key_id: None,
                api_key_id: Some("key-id-1"),
                prompt_tokens: None,
                completion_tokens: None,
                total_tokens: None,
                request_id: "req-fail",
                provider_request_id: None,
                served_by_model: None,
                routing_attempt_count: Some(1),
                routing_fallback_count: None,
                error_kind: Some("timeout"),
                error: Some("upstream request timed out after 7167ms"),
                mcp: None,
                cache: None,
                request_body_bytes: None,
                response_body_bytes: None,
            }
            .emit();
        });

        let out = writer.contents();
        assert!(out.contains("status=504"));
        // A call that never got a provider response must not carry an empty
        // `provider_request_id=""` — an always-present field defeats
        // filtering on it, same rule as `error_kind` above.
        assert!(!out.contains("provider_request_id"), "{out}");
        assert!(
            out.contains("error_kind=\"timeout\"") || out.contains("error_kind=timeout"),
            "{out}"
        );
        assert!(
            out.contains("upstream request timed out after 7167ms"),
            "{out}"
        );
    }

    #[test]
    fn emit_handles_missing_optional_fields() {
        let writer = VecWriter::default();
        let subscriber = fmt()
            .with_writer(writer.clone())
            .with_ansi(false)
            .with_target(false)
            .with_env_filter(EnvFilter::new("info"))
            .finish();

        with_default(subscriber, || {
            AccessLog {
                method: "POST",
                path: "/v1/chat/completions",
                status: 401,
                latency: Duration::from_millis(1),
                duration: Duration::from_millis(1),
                provider: None,
                model: None,
                upstream_model: None,
                provider_key_id: None,
                api_key_id: None,
                prompt_tokens: None,
                completion_tokens: None,
                total_tokens: None,
                request_id: "req-xyz",
                provider_request_id: None,
                served_by_model: None,
                routing_attempt_count: None,
                routing_fallback_count: None,
                error_kind: None,
                error: None,
                mcp: None,
                cache: None,
                request_body_bytes: None,
                response_body_bytes: None,
            }
            .emit();
        });
        let out = writer.contents();
        assert!(out.contains("status=401"));
        assert!(out.contains("proxy request completed"));
        // The fmt layer elides Option::None values; we should *not* see
        // a concrete provider rendered when the caller supplied None.
        assert!(!out.contains("provider=\"openai\""));
        // Same for the target pair (AISIX-Cloud#1571): a line written
        // before a target was selected must carry no target-derived field
        // at all, not an empty one an operator would have to filter out.
        assert!(!out.contains("upstream_model"), "{out}");
        assert!(!out.contains("provider_key_id"), "{out}");
        // An unknown body size is left off the line, never rendered as `0`,
        // which is a real size.
        assert!(!out.contains("body_bytes"), "{out}");
        // And no cache verdict: a line with no cache decision must not
        // claim one, since `cache_status` absent is how a reader tells
        // "this surface has no cache" from "the cache missed".
        assert!(!out.contains("cache_status"), "{out}");
    }

    /// AISIX-Cloud#1571: a response served out of the cache says so on the
    /// line. Without a marker there, the only evidence is the ABSENCE of
    /// target fields — which is also what a pre-dispatch failure looks
    /// like, so an operator cannot tell a Redis-served answer from a
    /// request that never reached a provider.
    #[test]
    fn emit_renders_the_cache_verdict() {
        let writer = VecWriter::default();
        let subscriber = fmt()
            .with_writer(writer.clone())
            .with_ansi(false)
            .with_target(false)
            .with_env_filter(EnvFilter::new("info"))
            .finish();

        with_default(subscriber, || {
            AccessLog {
                method: "POST",
                path: "/v1/chat/completions",
                status: 200,
                latency: Duration::from_millis(1),
                duration: Duration::from_millis(1),
                provider: Some("unknown"),
                model: Some("my-group"),
                upstream_model: None,
                provider_key_id: None,
                api_key_id: Some("key-id-1"),
                prompt_tokens: Some(2),
                completion_tokens: Some(1),
                total_tokens: Some(3),
                request_id: "req-cached",
                provider_request_id: None,
                served_by_model: None,
                routing_attempt_count: None,
                routing_fallback_count: None,
                error_kind: None,
                error: None,
                mcp: None,
                cache: Some(CacheAccessLog {
                    status: "hit",
                    hit_layer: Some("semantic"),
                }),
                request_body_bytes: None,
                response_body_bytes: None,
            }
            .emit();
        });
        let out = writer.contents();
        assert!(
            out.contains("cache_status=\"hit\"") || out.contains("cache_status=hit"),
            "{out}"
        );
        assert!(
            out.contains("cache_hit_layer=\"semantic\"")
                || out.contains("cache_hit_layer=semantic"),
            "{out}"
        );
    }

    /// A non-hit reports its status and no layer — an always-present
    /// `cache_hit_layer=""` would defeat filtering on it, the same rule
    /// `error_kind` and `provider_request_id` follow above.
    #[test]
    fn emit_omits_the_hit_layer_on_a_miss() {
        let writer = VecWriter::default();
        let subscriber = fmt()
            .with_writer(writer.clone())
            .with_ansi(false)
            .with_target(false)
            .with_env_filter(EnvFilter::new("info"))
            .finish();

        with_default(subscriber, || {
            AccessLog {
                method: "POST",
                path: "/v1/chat/completions",
                status: 200,
                latency: Duration::from_millis(1),
                duration: Duration::from_millis(1),
                provider: Some("openai"),
                model: Some("my-gpt4"),
                upstream_model: Some("gpt-4o"),
                provider_key_id: Some("pk-1"),
                api_key_id: Some("key-id-1"),
                prompt_tokens: Some(2),
                completion_tokens: Some(1),
                total_tokens: Some(3),
                request_id: "req-miss",
                provider_request_id: Some("chatcmpl-1"),
                served_by_model: None,
                routing_attempt_count: None,
                routing_fallback_count: None,
                error_kind: None,
                error: None,
                mcp: None,
                cache: Some(CacheAccessLog {
                    status: "miss",
                    hit_layer: None,
                }),
                request_body_bytes: None,
                response_body_bytes: None,
            }
            .emit();
        });
        let out = writer.contents();
        assert!(
            out.contains("cache_status=\"miss\"") || out.contains("cache_status=miss"),
            "{out}"
        );
        assert!(!out.contains("cache_hit_layer"), "{out}");
    }

    /// The semantic half is completed on the owned record, after the line
    /// was built — and only the halves the router actually decided are
    /// written: a line with no decision carries none of the three keys.
    #[test]
    fn record_writes_the_semantic_decision_it_was_completed_with() {
        let writer = VecWriter::default();
        let subscriber = fmt()
            .with_writer(writer.clone())
            .with_ansi(false)
            .with_target(false)
            .with_env_filter(EnvFilter::new("info"))
            .finish();
        let line = AccessLog {
            method: "POST",
            path: "/v1/chat/completions",
            status: 200,
            latency: Duration::from_millis(1),
            duration: Duration::from_millis(1),
            provider: None,
            model: Some("router"),
            upstream_model: None,
            provider_key_id: None,
            api_key_id: None,
            prompt_tokens: None,
            completion_tokens: None,
            total_tokens: None,
            request_id: "req-sem",
            provider_request_id: None,
            served_by_model: None,
            routing_attempt_count: None,
            routing_fallback_count: None,
            error_kind: None,
            error: None,
            mcp: None,
            cache: None,
            request_body_bytes: None,
            response_body_bytes: None,
        };
        with_default(subscriber, || {
            let mut matched = line.to_record();
            matched.semantic = Some(SemanticAccessLog {
                route: Some("code".into()),
                score: Some(0.81),
                fallback: None,
            });
            matched.emit();
            let mut fell_back = line.to_record();
            fell_back.semantic = Some(SemanticAccessLog {
                route: None,
                score: Some(0.2),
                fallback: Some(SemanticFallback::LowConfidence),
            });
            fell_back.emit();
            line.to_record().emit();
        });
        let out = writer.contents();
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines.len(), 3, "{out}");
        assert!(lines[0].contains("semantic_route=\"code\""), "{out}");
        assert!(lines[0].contains("semantic_score=0.81"), "{out}");
        assert!(!lines[0].contains("semantic_fallback"), "{out}");
        assert!(!lines[1].contains("semantic_route"), "{out}");
        assert!(lines[1].contains("semantic_score=0.2"), "{out}");
        assert!(
            lines[1].contains("semantic_fallback=\"low_confidence\""),
            "{out}"
        );
        assert!(!lines[2].contains("semantic_"), "{out}");
    }
}
