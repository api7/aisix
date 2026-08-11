//! aisix-proxy — client-facing proxy router (`:3000`).
//!
//! Mounts the OpenAI-compatible surface:
//! - `GET  /livez`
//! - `POST /v1/chat/completions` (streaming + non-streaming)
//!
//! Handlers run behind the [`AuthenticatedKey`] extractor which reads
//! the Bearer token (or `x-api-key` fallback) and looks the key up in
//! the current [`AisixSnapshot`]. Model authorisation is enforced per
//! request against `ApiKey::allowed_models`. Upstream calls are
//! dispatched through the [`aisix_gateway::Hub`] to the registered
//! `Bridge` for the Model's provider.
//!
//! Errors surface as OpenAI-style envelopes:
//!
//! ```json
//! {"error":{"message":"…","type":"…"}}
//! ```
//!
//! Status codes follow [`crate::error::ProxyError::status`] — spec §3 auth
//! rules (401/403), `Bridge` mapping preserves upstream 4xx and collapses
//! upstream 5xx to 502.

#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

mod a2a;
mod attempt;
mod audio;
mod auth;
pub mod background;
pub mod budget;
mod chat;
mod client_ip;
mod completions;
pub(crate) mod cooldown;
mod count_tokens;
mod dispatch;
mod ebml;
mod embeddings;
mod ensemble;
mod error;
mod error_translate;
pub mod health;
mod http_client;
mod images;
mod jobs;
mod jwt;
mod mcp;
mod messages;
mod model_resolve;
mod models;
mod passthrough;
mod quota;
mod realtime;
mod redact;
mod reject;
mod render;
mod request_id;
mod request_metrics;
mod rerank;
mod responses;
mod responses_bridge;
mod rewrite;
mod routing;
mod semantic;
pub mod sse_keepalive;
mod state;
mod stream_timeout;
mod token_estimate;
mod usage_attr;
mod util;
mod videos;

pub use auth::AuthenticatedKey;
pub use error::{ErrorEnvelope, ProxyError};
pub use health::{
    HealthTracker, LivezState, ModelRuntimeStatusTracker, RuntimeStatus, RuntimeStatusSnapshot,
};
pub use state::{CacheBackends, ProxyState};

use aisix_obs::AccessLog;
use axum::extract::State;
use axum::http::{header, HeaderValue, Request};
use axum::middleware::{self, Next};
use axum::response::Response;
use axum::routing::{any, get, post};
use axum::Router;
use tower_http::set_header::SetResponseHeaderLayer;

/// Product token emitted in the `Server` response header. Format follows
/// RFC 9110 §10.2.4 (`product/version`) and matches the convention used
/// by adjacent gateways (APISIX, nginx, kong). Version is
/// [`aisix_core::BUILD_VERSION`]: CI-stamped from the release tag, crate
/// version for local builds.
static SERVER_HEADER_VALUE: std::sync::LazyLock<HeaderValue> = std::sync::LazyLock::new(|| {
    HeaderValue::from_str(&format!("AISIX/{}", aisix_core::BUILD_VERSION))
        .expect("build version must be a valid ASCII header value")
});

/// Build the proxy router. Mounts `/livez` plus the
/// OpenAI-compatible proxy surface.
pub fn build_router(state: ProxyState) -> Router {
    let body_limit = state.request_body_limit_bytes;
    let router = Router::new()
        .route("/livez", get(livez))
        .route("/readyz", get(readyz))
        .route("/v1/models", get(models::list_models))
        .route("/v1/chat/completions", post(chat::chat_completions))
        .route("/v1/completions", post(completions::completions))
        .route("/v1/embeddings", post(embeddings::embeddings))
        .route("/v1/images/generations", post(images::image_generations))
        .route("/v1/messages", post(messages::messages))
        .route(
            "/v1/messages/count_tokens",
            post(count_tokens::count_tokens),
        )
        .route("/v1/rerank", post(rerank::rerank))
        .route("/v1/responses", post(responses::responses))
        .route("/v1/audio/transcriptions", post(audio::transcriptions))
        .route("/v1/audio/translations", post(audio::translations))
        .route("/v1/audio/speech", post(audio::speech))
        // Unified video-generation surface (AISIX-Cloud#1118 Phase 1):
        // submit → poll → fetch. Auth/ACL/quota enforced inside the
        // handlers; the GET routes are exempt from model-level rate
        // limits by design (see videos.rs).
        .route("/v1/videos", post(videos::create_video))
        .route("/v1/videos/:id", get(videos::get_video))
        .route("/v1/videos/:id/content", get(videos::video_content))
        // OpenAI Realtime WebSocket relay (#721). Auth/ACL/quota are
        // enforced pre-upgrade inside the handler.
        .route("/v1/realtime", get(realtime::realtime))
        // Files / Batches / Fine-tuning jobs surface (#720). Provider
        // routing rides the gateway-encoded resource ids; see jobs.rs.
        .route(
            "/v1/files",
            post(jobs::create_file).get(jobs::list_files),
        )
        .route(
            "/v1/files/:id",
            get(jobs::get_file).delete(jobs::delete_file),
        )
        .route("/v1/files/:id/content", get(jobs::file_content))
        .route(
            "/v1/batches",
            post(jobs::create_batch).get(jobs::list_batches),
        )
        .route("/v1/batches/:id", get(jobs::get_batch))
        .route("/v1/batches/:id/cancel", post(jobs::cancel_batch))
        .route(
            "/v1/fine_tuning/jobs",
            post(jobs::create_ft_job).get(jobs::list_ft_jobs),
        )
        .route("/v1/fine_tuning/jobs/:id", get(jobs::get_ft_job))
        .route(
            "/v1/fine_tuning/jobs/:id/cancel",
            post(jobs::cancel_ft_job),
        )
        .route(
            "/passthrough/:provider/*rest",
            any(passthrough::passthrough),
        )
        // Downstream-facing MCP gateway. Authentication (AISIX API key) is
        // enforced inside the handler via the `AuthenticatedKey` extractor.
        // `/mcp/{server}` is the single-server variant (original tool names);
        // the static `/mcp/` route wins over the param route for that path.
        .route("/mcp", any(mcp::mcp_endpoint))
        .route("/mcp/", any(mcp::mcp_endpoint))
        .route("/mcp/:server", any(mcp::mcp_scoped_endpoint))
        // Downstream-facing A2A gateway. One route per registered agent; the
        // agent's card (with the service URL rewritten to the gateway) is served
        // at the RFC 8615 well-known path under it. Authentication (AISIX API
        // key) is enforced inside the handlers via `AuthenticatedKey`.
        .route("/a2a/:agent", post(a2a::a2a_endpoint))
        .route(
            "/a2a/:agent/.well-known/agent-card.json",
            get(a2a::a2a_agent_card),
        )
        // Wire the configured cap into axum's request-body extractor
        // chain (`Json<T>` defers to `Bytes` which honors this layer).
        // Without this, axum 0.7's `DefaultBodyLimit` falls back to
        // its built-in 2 MiB default, which silently rejects bodies
        // in the 2 MiB-to-cap band with a stock `BytesRejection`
        // response (NOT the OpenAI envelope). The middleware below
        // catches the Content-Length-known case ahead of the
        // extractor; this layer catches chunked / size-mismatched
        // bodies once their actual byte count exceeds the cap.
        // `0` = no cap — `disable()` rather than omitting the layer,
        // because omitting it would fall back to axum's 2 MiB, not to
        // "unlimited".
        .layer(if body_limit > 0 {
            axum::extract::DefaultBodyLimit::max(body_limit)
        } else {
            axum::extract::DefaultBodyLimit::disable()
        })
        .layer(middleware::from_fn_with_state(
            state.clone(),
            enforce_request_body_limit,
        ))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            record_in_flight_request,
        ))
        // Record requests the caller abandoned before any response head
        // existed. Sits outside `record_in_flight_request` so a hang-up
        // during body upload (which the body-limit layers above are
        // awaiting) is captured too, and inside `ensure_request_id` so
        // the emitted line carries the same request id the caller was
        // handed. See `record_client_cancel`.
        .layer(middleware::from_fn_with_state(
            state.clone(),
            record_client_cancel,
        ))
        // Identify the data plane on every response, including error
        // envelopes and short-circuited responses from the layers
        // above. `overriding` (vs `if_not_present`) ensures the
        // gateway's identity is authoritative — any Server header set
        // by inner handlers or accidentally proxied from upstream is
        // replaced, so client-visible Server never leaks provider
        // identity.
        .layer(SetResponseHeaderLayer::overriding(
            header::SERVER,
            SERVER_HEADER_VALUE.clone(),
        ))
        .with_state(state.clone());

    // Pre-routing URL rewriting (`proxy.url_rewrites`). `Router::layer`
    // middleware runs AFTER route matching, so a URI rewritten there could
    // never change which route matches. Wrapping the whole router as the
    // fallback of an outer router — whose "routing" trivially resolves to
    // that fallback — gives the rewrite layer a genuine pre-routing seat:
    // it mutates the URI, then the inner router matches on the rewritten
    // path. Built only when rules are configured, so the default path pays
    // nothing. See rewrite.rs.
    let router = if state.url_rewrites.is_empty() {
        router
    } else {
        Router::new()
            .fallback_service(router)
            .layer(middleware::from_fn_with_state(
                state.clone(),
                rewrite::rewrite_request_uri,
            ))
    };

    // Outermost: mint the request id into the request extensions
    // before any handler/extractor runs — including the rewrite layer,
    // whose fired/failed log lines must carry the request span — and
    // stamp it onto every response (including the short-circuited 4xx
    // from the layers above) so the whole proxy family carries
    // `x-aisix-request-id` and it equals the telemetry request_id. See
    // request_id.rs.
    router.layer(middleware::from_fn_with_state(
        state,
        request_id::ensure_request_id,
    ))
}

async fn record_in_flight_request(
    State(state): State<ProxyState>,
    request: Request<axum::body::Body>,
    next: Next,
) -> Response {
    // Normalize to a bounded route template BEFORE using the path as a
    // metric label. This middleware runs before authentication and before
    // route matching, so the raw `request.uri().path()` is fully
    // attacker-controlled — the `/passthrough/:provider/*rest` wildcard
    // suffix (or any 404 path) would otherwise let an unauthenticated
    // caller mint unbounded Prometheus time series (#451).
    let endpoint = normalize_endpoint_label(request.uri().path());
    let inbound_protocol = inbound_protocol_for_endpoint(endpoint).to_string();
    let _guard = InFlightGuard::new(
        state.metrics.clone(),
        endpoint.to_string(),
        inbound_protocol,
    );
    next.run(request).await
}

/// Collapse a raw request path to a fixed route template so metric labels
/// stay bounded regardless of caller-supplied path segments. Keep this
/// allowlist in sync with the routes registered in `build_router`; any
/// unrecognized path (including unmatched 404s) maps to `"other"`.
fn normalize_endpoint_label(path: &str) -> &'static str {
    match path {
        "/livez" => "/livez",
        "/readyz" => "/readyz",
        "/v1/models" => "/v1/models",
        "/v1/chat/completions" => "/v1/chat/completions",
        "/v1/completions" => "/v1/completions",
        "/v1/embeddings" => "/v1/embeddings",
        "/v1/images/generations" => "/v1/images/generations",
        "/v1/messages" => "/v1/messages",
        "/v1/messages/count_tokens" => "/v1/messages/count_tokens",
        "/v1/rerank" => "/v1/rerank",
        "/v1/responses" => "/v1/responses",
        "/v1/audio/transcriptions" => "/v1/audio/transcriptions",
        "/v1/audio/translations" => "/v1/audio/translations",
        "/v1/audio/speech" => "/v1/audio/speech",
        "/v1/videos" => "/v1/videos",
        "/mcp" | "/mcp/" => "/mcp",
        "/v1/realtime" => "/v1/realtime",
        "/v1/files" => "/v1/files",
        "/v1/batches" => "/v1/batches",
        "/v1/fine_tuning/jobs" => "/v1/fine_tuning/jobs",
        // `/v1/videos/:id` and `/v1/videos/:id/content` collapse together:
        // the id is the only thing that varies and neither is worth its own
        // series.
        _ if path.starts_with("/v1/videos/") => "/v1/videos/:id",
        _ if path.starts_with("/v1/files/") => "/v1/files/:id",
        _ if path.starts_with("/v1/batches/") => "/v1/batches/:id",
        _ if path.starts_with("/v1/fine_tuning/jobs/") => "/v1/fine_tuning/jobs/:id",
        _ if path.starts_with("/mcp/") => "/mcp/{server}",
        _ if path.starts_with("/a2a/") => "/a2a",
        _ if path.starts_with("/passthrough/") => "/passthrough/:provider/*rest",
        _ => "other",
    }
}

/// Protocol family a route speaks, keyed off the normalized endpoint label.
/// Shared by the in-flight gauge and the detailed request families
/// (`request_metrics`) so the two can't disagree.
fn inbound_protocol_for_endpoint(endpoint: &str) -> &'static str {
    if endpoint == "/v1/messages" || endpoint == "/v1/messages/count_tokens" {
        "anthropic"
    } else if endpoint == "/mcp" || endpoint == "/mcp/{server}" {
        "mcp"
    } else if endpoint == "/a2a" {
        "a2a"
    } else if endpoint == "/v1/realtime" {
        "realtime"
    } else {
        "openai"
    }
}

struct InFlightGuard {
    metrics: std::sync::Arc<aisix_obs::Metrics>,
    endpoint: String,
    inbound_protocol: String,
}

impl InFlightGuard {
    fn new(
        metrics: std::sync::Arc<aisix_obs::Metrics>,
        endpoint: String,
        inbound_protocol: String,
    ) -> Self {
        metrics.increment_proxy_in_flight(&endpoint, &inbound_protocol);
        Self {
            metrics,
            endpoint,
            inbound_protocol,
        }
    }
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        self.metrics
            .decrement_proxy_in_flight(&self.endpoint, &self.inbound_protocol);
    }
}

/// nginx's non-standard "client closed request" status, used here purely
/// as a recorded outcome — nothing is ever sent to a caller that already
/// hung up. LiteLLM reports the same event as 499, so an operator running
/// both reads one number.
pub(crate) const CLIENT_CLOSED_REQUEST: u16 = 499;

/// `error_kind` for an abandoned request, mirroring LiteLLM's
/// `ClientDisconnected` error class.
const CLIENT_DISCONNECTED_KIND: &str = "client_disconnected";

/// Record a request whose caller hung up before the response head was
/// written.
///
/// Every endpoint logs and meters itself at the end of its own handler —
/// 29 `emit_access_log` call sites across 12 modules. When the client
/// disconnects first, axum drops the handler future and *none* of that
/// code runs: the request leaves no access-log line, no usage event and
/// no metric. It is invisible exactly where an operator most needs it,
/// because the usual reason a caller gives up is a long
/// time-to-first-token.
///
/// A cancelled future is only observable from `Drop`, so arm a guard,
/// disarm it once the inner service yields a response, and emit from
/// `Drop` when it is still armed. Doing it in one layer rather than in
/// each handler also keeps the endpoint family from drifting the way the
/// request-id header did before `ensure_request_id` (see request_id.rs).
///
/// This is NOT the streaming-disconnect path: once SSE bytes flow the
/// response head is already committed, so the handler has logged and the
/// per-stream `Drop` guard emits the usage event (see
/// `chat::build_sse_stream`). Response bodies are polled after this
/// middleware has returned, so a mid-stream hang-up leaves the guard
/// disarmed and is not double-counted here.
async fn record_client_cancel(
    State(state): State<ProxyState>,
    request: Request<axum::body::Body>,
    next: Next,
) -> Response {
    let mut guard = ClientCancelGuard {
        armed: true,
        metrics: state.metrics.clone(),
        endpoint: normalize_endpoint_label(request.uri().path()),
        method: request.method().clone(),
        uri: request.uri().clone(),
        request_id: request
            .extensions()
            .get::<request_id::RequestId>()
            .map(|id| id.0.clone())
            .unwrap_or_default(),
        started: std::time::Instant::now(),
    };
    let response = next.run(request).await;
    guard.armed = false;
    response
}

struct ClientCancelGuard {
    /// Cleared when the inner service returns. Still set at drop time
    /// means the future was cancelled rather than completed.
    armed: bool,
    metrics: std::sync::Arc<aisix_obs::Metrics>,
    /// Bounded route template — safe as a metric label (#451).
    endpoint: &'static str,
    method: axum::http::Method,
    /// `Uri` clones are reference-counted internally; the path is only
    /// read on the cancel path, so the happy path pays no formatting.
    uri: axum::http::Uri,
    request_id: String,
    started: std::time::Instant,
}

impl Drop for ClientCancelGuard {
    fn drop(&mut self) {
        // A panicking handler also leaves the guard armed — its future is
        // dropped mid-unwind exactly like a cancelled one. Attributing that
        // to the caller would be wrong twice over: it fabricates a client
        // disconnect that never happened, and it buries the panic under a
        // benign-looking 499. A panic has its own signal (tokio surfaces the
        // task failure and hyper drops the connection), so stay silent and
        // let that stand. Emitting here would also risk a double panic,
        // which aborts the process.
        if !self.armed || std::thread::panicking() {
            return;
        }
        let latency = self.started.elapsed();
        AccessLog {
            method: self.method.as_str(),
            path: self.uri.path(),
            status: CLIENT_CLOSED_REQUEST,
            latency,
            provider: None,
            model: None,
            api_key_id: None,
            prompt_tokens: None,
            completion_tokens: None,
            total_tokens: None,
            request_id: &self.request_id,
            served_by_model: None,
            routing_attempt_count: None,
            routing_fallback_count: None,
            error_kind: Some(CLIENT_DISCONNECTED_KIND),
            error: Some("client closed the request before the response head was written"),
        }
        .emit();
        self.metrics.record_client_cancelled(self.endpoint);
    }
}

/// Per RFC 9110 §15.5.14, a request body that exceeds the gateway's
/// configured `request_body_limit_bytes` must surface as a clean
/// `413 Content Too Large` response — NOT an `ECONNRESET` from a
/// mid-write socket close. This middleware inspects the inbound
/// `Content-Length` header before any handler runs and short-circuits
/// with the OpenAI-shape error envelope when the declared size
/// exceeds the cap.
///
/// Bodies sent with chunked transfer encoding (no Content-Length)
/// fall through to handler-level body extraction, which still
/// enforces the limit but with the slower fail mode (the read errors
/// once the cap is hit). Catching the Content-Length-known case here
/// is the load-bearing user-visible win: the OpenAI Node SDK and
/// `fetch` both set Content-Length for non-streamed POSTs, and
/// without this middleware they see ECONNRESET (indistinguishable
/// from a network failure or a gateway crash) instead of 413.
///
/// Both short-circuits answer through [`crate::reject`], so a request
/// refused here still produces the access-log line and request metrics
/// every other terminal path emits — the handler it never reached can't
/// do it. The logged latency spans the body drain below, which is what
/// an oversize request actually costs the gateway.
async fn enforce_request_body_limit(
    State(state): State<ProxyState>,
    request: Request<axum::body::Body>,
    next: Next,
) -> Response {
    let started = std::time::Instant::now();
    // /v1/messages must emit the Anthropic-shape error envelope
    // (closes #336). The middleware runs BEFORE the handler so the
    // handler's `into_anthropic_response()` would never see the
    // rejection — capture the inbound path here and use it to pick
    // the envelope shape on the reject paths below. Captured as
    // `bool` rather than holding a borrow into `request`, so the
    // `request.into_body()` move on the drain path doesn't conflict
    // with the captured value (audit HIGH-3 follow-up).
    //
    // Audit LOW-A (3rd audit): `/v1/messages/` (trailing slash) also
    // routes to the Anthropic handler via axum's path normalization,
    // but an exact-match check would miss it. The official Anthropic
    // SDK never appends a trailing slash so real-world exposure is
    // near-zero, but non-SDK callers (curl, custom clients) could
    // hit it. Accept both forms.
    let path = request.uri().path();
    let is_anthropic_path =
        path == "/v1/messages" || path == "/v1/messages/" || path == "/v1/messages/count_tokens";
    let envelope = if is_anthropic_path {
        reject::Envelope::Anthropic
    } else {
        reject::Envelope::OpenAi
    };
    // RFC 9110 §8.6 — a server SHOULD reject a request that carries
    // duplicate or conflicting `Content-Length` values rather than
    // act on the first one (which is a request-smuggling vector).
    let mut content_lengths = request
        .headers()
        .get_all(axum::http::header::CONTENT_LENGTH)
        .iter();
    // The access log takes the bounded route template, not the raw path.
    // This is the one logging site that runs before authentication AND
    // before route matching, so the raw path is unvalidated
    // caller-controlled text — the same reason the metric labels are
    // collapsed (#451).
    let endpoint = normalize_endpoint_label(path);
    let first = content_lengths.next();
    if content_lengths.next().is_some() {
        return reject::reject_before_dispatch(
            &state,
            request.method().as_str(),
            endpoint,
            &request_id_of(&request),
            None,
            started,
            envelope,
            ProxyError::InvalidRequest("conflicting Content-Length headers".into()),
        );
    }
    // `0` = the cap is disabled; the duplicate-Content-Length rejection
    // above still applies — that one is request-smuggling hygiene, not a
    // size limit.
    if let Some(declared) = first
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<usize>().ok())
    {
        if state.request_body_limit_bytes > 0 && declared > state.request_body_limit_bytes {
            // Capture what the logs need before the body move below
            // consumes the request.
            let method = request.method().clone();
            let request_id = request_id_of(&request);
            // Drain the inbound body so hyper can flush the 413 response
            // on the same HTTP/1.1 connection. Without this, hyper closes
            // the socket while the client is still writing, and the client
            // sees EPIPE/ECONNRESET instead of the 413.
            let drain = drain_body(request.into_body()).await;
            record_body_limit_rejection(
                &state,
                endpoint,
                method.as_str(),
                &request_id,
                declared,
                &drain,
                started,
            );
            return reject::reject_before_dispatch(
                &state,
                method.as_str(),
                endpoint,
                &request_id,
                None,
                started,
                envelope,
                ProxyError::RequestTooLarge {
                    limit_bytes: state.request_body_limit_bytes,
                },
            );
        }
    }
    next.run(request).await
}

/// The id `ensure_request_id` (the outermost layer) minted for this
/// request, so a rejection logged here joins the `x-aisix-request-id` the
/// caller was handed. The fallback only covers a router assembled without
/// that layer — every shipped path has it.
fn request_id_of(request: &Request<axum::body::Body>) -> String {
    request
        .extensions()
        .get::<request_id::RequestId>()
        .map(|r| r.0.clone())
        .unwrap_or_else(request_id::new_request_id)
}

/// How the drain of a refused body ended. A fixed vocabulary: it is both
/// a metric label and a log field, and `completed` vs the rest is what
/// tells an operator whether the caller could still read the 413.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DrainOutcome {
    /// The caller finished sending. The connection is clean, so the 413
    /// reaches it.
    Completed,
    /// The byte cap was hit first. The rest of the body is never read, so
    /// the caller may see a reset instead of the 413.
    CapReached,
    /// The time budget expired with the caller still sending — same
    /// visible result as `CapReached`, different cause (a slow or
    /// dribbling client rather than a huge one).
    Timeout,
    /// The caller's stream errored mid-drain: it went away on its own.
    ClientReadError,
}

impl DrainOutcome {
    fn as_str(self) -> &'static str {
        match self {
            DrainOutcome::Completed => "completed",
            DrainOutcome::CapReached => "cap_reached",
            DrainOutcome::Timeout => "timeout",
            DrainOutcome::ClientReadError => "client_read_error",
        }
    }
}

/// What a drain cost and how it ended.
struct Drain {
    bytes: usize,
    outcome: DrainOutcome,
}

/// Read and discard the inbound body, bounded by both bytes and time.
///
/// Byte cap (32 MiB) prevents a huge `Content-Length` from consuming
/// unbounded memory.  Time cap (5 s) prevents a slowloris-style
/// client from holding the task indefinitely by dribbling data.
///
/// Returns how it ended, because those two bounds are exactly the cases
/// where the caller gets a connection reset instead of the 413 the drain
/// exists to deliver — and a discarded result made a reset caused here
/// indistinguishable from one caused anywhere else.
async fn drain_body(body: axum::body::Body) -> Drain {
    use http_body_util::BodyExt;

    const DRAIN_CAP: usize = 32 * 1024 * 1024;
    const DRAIN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

    let mut drained = 0usize;
    // Bound the future to its own statement: holding it in the `match`
    // scrutinee would keep `drained` mutably borrowed through the arms.
    let ended = tokio::time::timeout(DRAIN_TIMEOUT, async {
        let mut body = body;
        loop {
            match body.frame().await {
                Some(Ok(frame)) => {
                    if let Some(data) = frame.data_ref() {
                        drained += data.len();
                        if drained >= DRAIN_CAP {
                            return DrainOutcome::CapReached;
                        }
                    }
                }
                // Distinguishing this from the clean end below is the
                // point: `while let Some(Ok(_))` folded a client that
                // vanished mid-body into "drained fine".
                Some(Err(_)) => return DrainOutcome::ClientReadError,
                None => return DrainOutcome::Completed,
            }
        }
    })
    .await;
    Drain {
        // On timeout the inner future is dropped mid-poll, so this is
        // what had been absorbed by then — the useful number.
        bytes: drained,
        outcome: ended.unwrap_or(DrainOutcome::Timeout),
    }
}

/// Emit the body-limit diagnostic for a refused request: what the caller
/// declared, what the cap was, how much of the body the gateway absorbed
/// and how that drain ended. Joined to the access log by `request_id`.
///
/// Level follows the outcome. A clean drain is routine and stays `info`;
/// a cap hit, a timeout or a mid-drain read error all mean the gateway
/// stopped reading while the caller was still writing — the case where
/// the caller sees a reset rather than the 413 — so those warn. The warn
/// is rate limited per outcome so a flood of oversize requests can't
/// amplify into a log flood; the counter above it is unconditional and
/// keeps the true volume regardless of what the limiter drops.
fn record_body_limit_rejection(
    state: &ProxyState,
    endpoint: &'static str,
    method: &str,
    request_id: &str,
    declared_content_length: usize,
    drain: &Drain,
    started: std::time::Instant,
) {
    let inbound_protocol = inbound_protocol_for_endpoint(endpoint);
    state
        .metrics
        .record_body_limit_rejection(endpoint, inbound_protocol, drain.outcome.as_str());

    let elapsed_ms = started.elapsed().as_millis() as u64;
    if drain.outcome == DrainOutcome::Completed {
        tracing::info!(
            target: "aisix::body_limit",
            request_id,
            endpoint,
            method,
            status = 413,
            declared_content_length,
            configured_limit_bytes = state.request_body_limit_bytes,
            drained_bytes = drain.bytes,
            drain_outcome = drain.outcome.as_str(),
            elapsed_ms,
            "request body exceeded the configured limit",
        );
    } else if warn_allowed(drain.outcome) {
        tracing::warn!(
            target: "aisix::body_limit",
            request_id,
            endpoint,
            method,
            status = 413,
            declared_content_length,
            configured_limit_bytes = state.request_body_limit_bytes,
            drained_bytes = drain.bytes,
            drain_outcome = drain.outcome.as_str(),
            elapsed_ms,
            "request body exceeded the configured limit and the drain did not finish",
        );
    }
}

/// At most one warn per abnormal outcome per second. Per outcome rather
/// than global so a steady stream of one kind can't mask the first of
/// another. Races between threads at the boundary can let a second line
/// through; that is cheaper than the synchronisation to prevent it, and
/// the counter is the source of truth for volume anyway.
fn warn_allowed(outcome: DrainOutcome) -> bool {
    use std::sync::atomic::{AtomicU64, Ordering};

    const WARN_INTERVAL_MS: u64 = 1_000;
    // `0` = never warned. Elapsed millis are clamped to >= 1 on store so
    // the sentinel can't collide with a warn in the first millisecond.
    static LAST_WARN_MS: [AtomicU64; 3] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
    static PROCESS_START: std::sync::LazyLock<std::time::Instant> =
        std::sync::LazyLock::new(std::time::Instant::now);

    let slot = match outcome {
        DrainOutcome::CapReached => 0,
        DrainOutcome::Timeout => 1,
        DrainOutcome::ClientReadError => 2,
        DrainOutcome::Completed => return false,
    };
    let now = (PROCESS_START.elapsed().as_millis() as u64).max(1);
    let last = LAST_WARN_MS[slot].load(Ordering::Relaxed);
    if last != 0 && now.saturating_sub(last) < WARN_INTERVAL_MS {
        return false;
    }
    LAST_WARN_MS[slot].store(now, Ordering::Relaxed);
    true
}

async fn livez(
    State(state): State<ProxyState>,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Response {
    crate::health::livez_response(&state.livez, params.contains_key("verbose"))
}

async fn readyz(
    State(state): State<ProxyState>,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Response {
    let config_block = state
        .config_apply_age
        .as_ref()
        .and_then(|probe| crate::health::config_readiness_block(probe()));
    crate::health::readyz_response(&state.livez, config_block, params.contains_key("verbose"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_label_is_bounded_for_arbitrary_paths() {
        // Known routes pass through unchanged.
        assert_eq!(
            normalize_endpoint_label("/v1/chat/completions"),
            "/v1/chat/completions"
        );
        assert_eq!(normalize_endpoint_label("/v1/messages"), "/v1/messages");
        // Passthrough collapses regardless of provider/suffix — this is the
        // unbounded-cardinality vector from #451.
        assert_eq!(
            normalize_endpoint_label("/passthrough/openai/anything/unique-123"),
            "/passthrough/:provider/*rest"
        );
        assert_eq!(
            normalize_endpoint_label("/passthrough/openai/other-unique-456"),
            "/passthrough/:provider/*rest"
        );
        // Arbitrary unauthenticated paths bucket to a single label.
        assert_eq!(normalize_endpoint_label("/random/x"), "other");
        assert_eq!(normalize_endpoint_label("/random/y"), "other");
        // The scoped MCP endpoint collapses to one label regardless of the
        // server segment; the aggregated endpoint (with and without the
        // trailing slash) keeps its own — the exact `"/mcp/"` arm must stay
        // ahead of the `/mcp/` prefix arm.
        assert_eq!(normalize_endpoint_label("/mcp/alpha"), "/mcp/{server}");
        assert_eq!(normalize_endpoint_label("/mcp/unique-xyz"), "/mcp/{server}");
        assert_eq!(normalize_endpoint_label("/mcp"), "/mcp");
        assert_eq!(normalize_endpoint_label("/mcp/"), "/mcp");
    }

    use aisix_core::resource::ResourceEntry;
    use aisix_core::snapshot::SnapshotHandle;
    use aisix_core::{AisixSnapshot, ApiKey, Model, ProxyConfig};
    use aisix_gateway::{Hub, SseDecoder, SseEvent};
    use aisix_provider_openai::OpenAiBridge;
    use axum::body::{to_bytes, Body};
    use axum::http::{Request, StatusCode};
    use futures::StreamExt;
    use reqwest::Client;
    use std::sync::Arc;
    use tower::ServiceExt;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn cfg() -> ProxyConfig {
        ProxyConfig {
            addr: "127.0.0.1:0".into(),
            request_body_limit_bytes: 1_048_576,
            real_ip: Default::default(),
            request_id: Default::default(),
            url_rewrites: Vec::new(),
            tls: None,
            thread_per_core: None,
            workers: None,
        }
    }

    fn openai_test_bridge() -> OpenAiBridge {
        let client = Client::builder()
            .user_agent("aisix-test/0.1")
            .no_proxy()
            .build()
            .unwrap();
        OpenAiBridge::with_client(client)
    }

    /// State used by the *existing* tests — cache disabled so the
    /// rate-limit / wiremock cases still see every request reach the
    /// upstream. Cache-specific tests build state with the default
    /// constructor (which keeps caching on) instead.
    fn build_state(snapshot: AisixSnapshot, hub: Arc<Hub>) -> ProxyState {
        let handle = SnapshotHandle::new(snapshot);
        ProxyState::new(handle, hub, &cfg()).without_cache()
    }

    fn build_state_with_cache(snapshot: AisixSnapshot, hub: Arc<Hub>) -> ProxyState {
        let handle = SnapshotHandle::new(snapshot);
        ProxyState::new(handle, hub, &cfg())
    }

    const PK_ID: &str = "11111111-1111-1111-1111-111111111111";

    fn model_entry(name: &str) -> ResourceEntry<Model> {
        let cfg = format!(
            r#"{{
                "display_name": "{name}",
                "provider": "openai",
                "model_name": "gpt-4o",
                "provider_key_id": "{PK_ID}"
            }}"#
        );
        let model: Model = serde_json::from_str(&cfg).unwrap();
        ResourceEntry::new("model-id-1", model, 1)
    }

    fn provider_key_entry(api_base: &str) -> ResourceEntry<aisix_core::ProviderKey> {
        let cfg = format!(
            r#"{{"display_name":"openai-up","secret":"sk-upstream","api_base":"{api_base}","provider":"openai","adapter":"openai"}}"#
        );
        let pk: aisix_core::ProviderKey = serde_json::from_str(&cfg).unwrap();
        ResourceEntry::new(PK_ID, pk, 1)
    }

    fn new_snap(api_base: &str) -> AisixSnapshot {
        let snap = AisixSnapshot::new();
        snap.provider_keys.insert(provider_key_entry(api_base));
        snap
    }

    fn apikey_entry(key: &str, allowed: &[&str]) -> ResourceEntry<ApiKey> {
        apikey_entry_with_limits(key, allowed, None)
    }

    fn apikey_entry_with_limits(
        key: &str,
        allowed: &[&str],
        rate_limit: Option<serde_json::Value>,
    ) -> ResourceEntry<ApiKey> {
        let allowed_json = serde_json::to_string(&allowed).unwrap();
        let rl_tail = match rate_limit {
            Some(v) => format!(", \"rate_limit\": {v}"),
            None => String::new(),
        };
        // Tests pass the plaintext bearer here (e.g. "sk-caller"); the
        // wire schema stores its SHA-256 (§9A.7B.4). Hash via the
        // canonical helper so request-side `Bearer <plaintext>` lookups
        // line up.
        let key_hash = ApiKey::hash_bearer(key);
        let cfg =
            format!(r#"{{"key_hash": "{key_hash}", "allowed_models": {allowed_json}{rl_tail}}}"#);
        let apikey: ApiKey = serde_json::from_str(&cfg).unwrap();
        ResourceEntry::new("key-id-1", apikey, 1)
    }

    fn seed_snapshot(model: &str, allowed: &[&str], api_base: &str) -> AisixSnapshot {
        let snap = new_snap(api_base);
        snap.models.insert(model_entry(model));
        snap.apikeys.insert(apikey_entry("sk-caller", allowed));
        snap
    }

    /// Seed an env-scope keyword guardrail into a live snapshot handle.
    ///
    /// Uses `handle.rcu()` to atomically replace the snapshot and bump the
    /// version counter. `LiveGuardrailIndex` compares versions on every
    /// `resolve()` call; without the bump it would return a stale (empty)
    /// index regardless of when this helper is called relative to
    /// `build_state`. With the bump the index rebuilds on the next request,
    /// making the call-order invariant.
    ///
    /// `guardrail_json` must be a valid inline `Guardrail` JSON payload
    /// (same wire shape as `/aisix/<env>/guardrails/<uuid>`).
    /// A single env-scope attachment is inserted alongside it so the
    /// guardrail fires on every request.
    fn seed_guardrail(
        handle: &SnapshotHandle<AisixSnapshot>,
        guardrail_id: &str,
        guardrail_json: &str,
    ) {
        use aisix_core::models::{Guardrail as DomainGuardrail, GuardrailAttachment};
        let gid = guardrail_id.to_string();
        let row: DomainGuardrail = serde_json::from_str(guardrail_json).unwrap();
        let att: GuardrailAttachment = serde_json::from_str(&format!(
            r#"{{"guardrail_id": "{gid}", "scope_type": "env", "priority": 50}}"#
        ))
        .unwrap();
        // rcu: load current snapshot → clone it → insert guardrail entries →
        // store the new snapshot and bump the version. The closure is
        // idempotent: re-inserting the same id merely overwrites with
        // identical data, so retries under contention are safe.
        handle.rcu(|snap| {
            let new_snap = snap.clone();
            new_snap
                .guardrails
                .insert(ResourceEntry::new(gid.clone(), row.clone(), 1));
            new_snap.guardrail_attachments.insert(ResourceEntry::new(
                format!("att-{gid}"),
                att.clone(),
                1,
            ));
            new_snap
        });
    }

    /// Insert a default-enabled cache policy on the snapshot so the
    /// proxy's cache gate (chat::dispatch) opens the lookup path.
    /// Stage 2 honors existence + `enabled`; Stage 3 honors
    /// `applies_to`. The default `applies_to=all` (set by serde
    /// when omitted) matches every request, so existing tests that
    /// seed a bare policy keep passing.
    fn seed_cache_policy(snap: &AisixSnapshot, name: &str) {
        seed_cache_policy_with_applies_to(snap, name, "all");
    }

    /// Like `seed_cache_policy` but with a specific `applies_to`
    /// clause — used by the Stage 3 tests that pin the matcher's
    /// behaviour on `model:<name>` / `api_key:<id>`.
    fn seed_cache_policy_with_applies_to(snap: &AisixSnapshot, name: &str, applies_to: &str) {
        let cfg =
            format!(r#"{{"name": "{name}", "backend": "memory", "applies_to": "{applies_to}"}}"#,);
        let policy: aisix_core::models::CachePolicy = serde_json::from_str(&cfg).unwrap();
        snap.cache_policies
            .insert(ResourceEntry::new(format!("cp-id-{name}"), policy, 1));
    }

    /// Disabled-policy seeder for #154 regression coverage. Posts a
    /// `CachePolicy{enabled: false, applies_to: "all"}` so the
    /// cache-gate predicate at chat.rs (`entry.value.enabled && ...`)
    /// must skip it.
    fn seed_cache_policy_disabled(snap: &AisixSnapshot, name: &str) {
        let cfg = format!(
            r#"{{"name": "{name}", "backend": "memory", "applies_to": "all", "enabled": false}}"#,
        );
        let policy: aisix_core::models::CachePolicy = serde_json::from_str(&cfg).unwrap();
        snap.cache_policies
            .insert(ResourceEntry::new(format!("cp-id-{name}"), policy, 1));
    }

    /// Policy seeder with an explicit `backend` — used by the #519
    /// B.8 tests that pin per-policy backend dispatch.
    fn seed_cache_policy_with_backend(snap: &AisixSnapshot, name: &str, backend: &str) {
        let cfg = format!(r#"{{"name": "{name}", "backend": "{backend}", "applies_to": "all"}}"#);
        let policy: aisix_core::models::CachePolicy = serde_json::from_str(&cfg).unwrap();
        snap.cache_policies
            .insert(ResourceEntry::new(format!("cp-id-{name}"), policy, 1));
    }

    fn seed_snapshot_with_limits(
        model: &str,
        allowed: &[&str],
        api_base: &str,
        rate_limit: serde_json::Value,
    ) -> AisixSnapshot {
        let snap = new_snap(api_base);
        snap.models.insert(model_entry(model));
        snap.apikeys.insert(apikey_entry_with_limits(
            "sk-caller",
            allowed,
            Some(rate_limit),
        ));
        snap
    }

    async fn run(app: Router, req: Request<Body>) -> axum::http::Response<Body> {
        app.oneshot(req).await.unwrap()
    }

    #[tokio::test]
    async fn non_streaming_happy_path_returns_openai_shaped_json() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-upstream",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}
            })))
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        let app = build_router(build_state(snap, hub));

        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "hello"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["object"], "chat.completion");
        assert_eq!(v["choices"][0]["message"]["content"], "hi");
        assert_eq!(v["usage"]["total_tokens"], 3);
    }

    #[tokio::test]
    async fn missing_authorization_returns_401_envelope() {
        let hub = Arc::new(Hub::new());
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], "http://unused");
        let app = build_router(build_state(snap, hub));

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"model":"my-gpt4","messages":[]}"#))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "invalid_api_key");
    }

    #[tokio::test]
    async fn livez_reports_plain_ok_by_default() {
        let hub = Arc::new(Hub::new());
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], "http://unused");
        let app = build_router(build_state(snap, hub));

        let req = Request::builder()
            .method("GET")
            .uri("/livez")
            .body(Body::empty())
            .unwrap();

        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        assert_eq!(std::str::from_utf8(&bytes).unwrap(), "ok");
    }

    /// Every response — including success bodies, error envelopes, and
    /// short-circuited middleware rejections — must carry the gateway's
    /// `Server` product token (`AISIX/<semver>`) so clients can identify
    /// the data plane without round-tripping to a status endpoint.
    #[tokio::test]
    async fn server_header_identifies_the_data_plane() {
        let hub = Arc::new(Hub::new());
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], "http://unused");
        let app = build_router(build_state(snap, hub));

        // Success path — plain handler response.
        let ok_req = Request::builder()
            .method("GET")
            .uri("/livez")
            .body(Body::empty())
            .unwrap();
        let ok_resp = run(app.clone(), ok_req).await;
        let ok_server = ok_resp
            .headers()
            .get(axum::http::header::SERVER)
            .expect("success response must carry Server header")
            .to_str()
            .unwrap()
            .to_string();
        assert!(
            ok_server.starts_with("AISIX/") && ok_server.len() > "AISIX/".len(),
            "expected `AISIX/<version>`, got {ok_server:?}"
        );

        // Error path — auth failure envelope. Same Server header must
        // appear so error responses don't accidentally hide the gateway's
        // identity (and don't leak any upstream Server token).
        let unauth_req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"model":"my-gpt4","messages":[]}"#))
            .unwrap();
        let unauth_resp = run(app, unauth_req).await;
        assert_eq!(unauth_resp.status(), StatusCode::UNAUTHORIZED);
        let err_server = unauth_resp
            .headers()
            .get(axum::http::header::SERVER)
            .expect("error response must carry Server header")
            .to_str()
            .unwrap();
        assert_eq!(err_server, ok_server);
    }

    /// The 413 short-circuit runs INSIDE `SetResponseHeaderLayer` —
    /// `enforce_request_body_limit` rejects the request before any
    /// handler executes. This pins layer ordering: a regression that
    /// moves `SetResponseHeaderLayer` inside the body-limit middleware
    /// (or anywhere "below" it in the stack) would silently strip the
    /// Server header from 413 responses while every existing test
    /// still passed.
    #[tokio::test]
    async fn server_header_present_on_413_short_circuit() {
        let hub = Arc::new(Hub::new());
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], "http://unused");
        let app = build_router(build_state(snap, hub));

        let oversized = 2 * 1024 * 1024; // 2 MiB > 1 MiB cap from cfg()
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .header("content-length", oversized.to_string())
            .body(Body::from(
                r#"{"model":"my-gpt4","messages":[{"role":"user","content":"hi"}]}"#,
            ))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
        let server = resp
            .headers()
            .get(axum::http::header::SERVER)
            .expect("413 short-circuit must carry Server header")
            .to_str()
            .unwrap();
        assert!(
            server.starts_with("AISIX/"),
            "expected `AISIX/<version>`, got {server:?}"
        );
    }

    /// Security contract: the gateway must NEVER leak an upstream
    /// provider's `Server` token to the client. The passthrough handler
    /// copies upstream response headers wholesale (minus hop-by-hop),
    /// so an upstream like Cloudflare/nginx/gunicorn would surface its
    /// own Server unless `overriding` actually replaces it. A regression
    /// that swapped `overriding` → `if_not_present` would be a silent
    /// information-disclosure bug (provider fingerprinting via error
    /// envelopes) — this test locks the no-leak property.
    #[tokio::test]
    async fn server_header_overrides_upstream_provider_token_no_leak() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    // Upstream's identity — MUST NOT survive the round-trip.
                    .insert_header("server", "cloudflare-nginx/3.7-leakthis")
                    .set_body_json(serde_json::json!({
                        "id": "cmpl-upstream",
                        "model": "gpt-4o",
                        "choices": [{
                            "index": 0,
                            "message": {"role": "assistant", "content": "x"},
                            "finish_reason": "stop"
                        }],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
                    })),
            )
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        let app = build_router(build_state(snap, hub));

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"my-gpt4","messages":[{"role":"user","content":"hi"}]}"#,
            ))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);

        // Exactly one Server value — `overriding` replaces, doesn't append.
        let all: Vec<_> = resp
            .headers()
            .get_all(axum::http::header::SERVER)
            .iter()
            .collect();
        assert_eq!(
            all.len(),
            1,
            "exactly one Server value expected; got {all:?}"
        );

        let server = all[0].to_str().unwrap();
        assert!(
            server.starts_with("AISIX/"),
            "Server must be the gateway identity; got {server:?}"
        );
        assert!(
            !server.contains("cloudflare") && !server.contains("nginx"),
            "Upstream Server token leaked through; got {server:?}"
        );
    }

    #[tokio::test]
    async fn livez_rejects_non_get_requests() {
        let hub = Arc::new(Hub::new());
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], "http://unused");
        let app = build_router(build_state(snap, hub));

        let req = Request::builder()
            .method("POST")
            .uri("/livez")
            .body(Body::empty())
            .unwrap();

        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::METHOD_NOT_ALLOWED);
    }

    #[tokio::test]
    async fn livez_returns_503_when_shutting_down() {
        let hub = Arc::new(Hub::new());
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], "http://unused");
        let state = build_state(snap, hub);
        state.livez.mark_shutting_down();
        let app = build_router(state);

        let req = Request::builder()
            .method("GET")
            .uri("/livez")
            .body(Body::empty())
            .unwrap();

        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let text = std::str::from_utf8(&bytes).unwrap();
        assert!(text.contains("livez check failed"));
    }

    #[tokio::test]
    async fn readyz_503s_when_the_config_probe_reports_no_apply_yet() {
        // Pins the config_apply_age plumbing through the router: a wired
        // probe reporting "no apply yet" must gate readiness with a 503.
        // Without this, dropping the probe wiring would leave readyz
        // reporting `[+]config ok` unconditionally (the field-None path),
        // byte-identical to wired-and-fresh — a silent downgrade of
        // readiness to shutdown-only that no other test would notice.
        let hub = Arc::new(Hub::new());
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], "http://unused");
        let state = build_state(snap, hub).with_config_apply_age(Arc::new(|| None));
        let app = build_router(state);

        let req = Request::builder()
            .method("GET")
            .uri("/readyz?verbose")
            .body(Body::empty())
            .unwrap();

        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let text = std::str::from_utf8(&bytes).unwrap();
        assert!(text.contains("[-]config failed: not ready"));
    }

    #[tokio::test]
    async fn readyz_stays_200_when_no_config_event_has_arrived_in_hours() {
        // A gateway whose environment is not changing receives no config
        // events, so the apply age grows without bound while the gateway is
        // perfectly healthy. Readiness must not read that as a fault: it
        // used to 503 past five minutes, which emptied the Kubernetes
        // Service of every replica once a deployment went idle.
        let hub = Arc::new(Hub::new());
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], "http://unused");
        let state = build_state(snap, hub)
            .with_config_apply_age(Arc::new(|| Some(std::time::Duration::from_secs(7200))));
        let app = build_router(state);

        let req = Request::builder()
            .method("GET")
            .uri("/readyz?verbose")
            .body(Body::empty())
            .unwrap();

        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let text = std::str::from_utf8(&bytes).unwrap();
        assert!(text.contains("[+]config ok"));
    }

    #[tokio::test]
    async fn health_route_is_not_found() {
        let hub = Arc::new(Hub::new());
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], "http://unused");
        let app = build_router(build_state(snap, hub));

        let req = Request::builder()
            .method("GET")
            .uri("/health")
            .body(Body::empty())
            .unwrap();

        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn unknown_api_key_returns_401() {
        let hub = Arc::new(Hub::new());
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], "http://unused");
        let app = build_router(build_state(snap, hub));

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-does-not-exist")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"model":"my-gpt4","messages":[]}"#))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn model_not_in_allowed_list_returns_403() {
        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        // ApiKey allows only "other-model", the caller asks for "my-gpt4".
        let snap = seed_snapshot("my-gpt4", &["other-model"], "http://unused");
        let app = build_router(build_state(snap, hub));

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"my-gpt4","messages":[{"role":"user","content":"hi"}]}"#,
            ))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "permission_denied");
    }

    #[tokio::test]
    async fn unknown_model_returns_404_envelope() {
        let hub = Arc::new(Hub::new());
        let snap = seed_snapshot("my-gpt4", &["*"], "http://unused");
        let app = build_router(build_state(snap, hub));

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"no-such-model","messages":[{"role":"user","content":"hi"}]}"#,
            ))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "model_not_found");
    }

    #[tokio::test]
    async fn empty_messages_returns_400() {
        let hub = Arc::new(Hub::new());
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], "http://unused");
        let app = build_router(build_state(snap, hub));

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"model":"my-gpt4","messages":[]}"#))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    /// Issue #324: missing required field on the chat-completion
    /// request body must surface as **400 Bad Request** per OpenAI's
    /// wire contract, not 422 Unprocessable Entity. SDKs branching
    /// on the status code see different semantics depending on
    /// which proxy they sit behind; a customer migrating between
    /// OpenAI direct and a gateway-fronted deployment needs the
    /// 400-vs-422 distinction to be wire-stable.
    ///
    /// Pre-fix: axum's `Json<ChatFormat>` extractor returned
    /// `JsonRejection::JsonDataError` → 422.
    /// Post-fix: the handler intercepts the JsonRejection and maps
    /// to `ProxyError::InvalidRequest` → 400.
    #[tokio::test]
    async fn missing_model_field_returns_400_not_422() {
        let hub = Arc::new(Hub::new());
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], "http://unused");
        let app = build_router(build_state(snap, hub));

        // Valid JSON, valid `messages` field, but `model` omitted —
        // the OpenAI ChatCompletion contract requires it.
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"messages":[{"role":"user","content":"hi"}]}"#,
            ))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(
            resp.status(),
            StatusCode::BAD_REQUEST,
            "missing model field must surface as 400 per OpenAI wire contract — #324",
        );
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "invalid_request_error");
    }

    /// Companion case: missing `messages` field must also surface
    /// as 400. Same OpenAI wire contract — `messages` is required.
    #[tokio::test]
    async fn missing_messages_field_returns_400_not_422() {
        let hub = Arc::new(Hub::new());
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], "http://unused");
        let app = build_router(build_state(snap, hub));

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"model":"my-gpt4"}"#))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(
            resp.status(),
            StatusCode::BAD_REQUEST,
            "missing messages field must surface as 400 per OpenAI wire contract — #324",
        );
        // Pin the envelope shape too — a future regression that
        // returned 400 with a non-OpenAI envelope (or empty body)
        // would otherwise pass on status alone. Per audit MEDIUM on
        // PR #400.
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "invalid_request_error");
    }

    /// Companion case: malformed JSON (syntax error) also must
    /// surface as 400, not 422. Same handler path as #324 — the
    /// JsonRejection variants for syntax vs data error both map
    /// to InvalidRequest.
    #[tokio::test]
    async fn malformed_json_returns_400() {
        let hub = Arc::new(Hub::new());
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], "http://unused");
        let app = build_router(build_state(snap, hub));

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(r#"{not even valid json"#))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(
            resp.status(),
            StatusCode::BAD_REQUEST,
            "malformed JSON must surface as 400, not 422",
        );
        // Envelope-shape pin matching the sibling missing-field
        // tests — same JsonRejection → InvalidRequest path; the
        // envelope must stay OpenAI-shape on every variant.
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "invalid_request_error");
    }

    /// Issue #159: a request body whose declared `Content-Length`
    /// exceeds the configured cap must surface as `413 Content Too
    /// Large` per RFC 9110 §15.5.14, NOT as `ECONNRESET` from a
    /// mid-write socket close. Regression: the handler-level body
    /// extractor's overflow path was racing the client write,
    /// surfacing as a network failure indistinguishable from a
    /// gateway crash. The new middleware short-circuits on the
    /// declared size before any handler runs.
    #[tokio::test]
    async fn oversize_body_returns_413_envelope_with_content_length_check() {
        let hub = Arc::new(Hub::new());
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], "http://unused");
        let app = build_router(build_state(snap, hub));

        // Declare a Content-Length well over the test cfg's 1 MiB cap
        // but ship a tiny actual body — the middleware MUST reject
        // based on the declared size alone, before reading the body.
        // (A real caller's `JSON.stringify` would set Content-Length
        // matching the body size; the assertion is "we trust the
        // declared header for the early reject".)
        let oversized = 2 * 1024 * 1024; // 2 MiB > 1 MiB cap
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .header("content-length", oversized.to_string())
            .body(Body::from(
                r#"{"model":"my-gpt4","messages":[{"role":"user","content":"hi"}]}"#,
            ))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        // OpenAI-shape envelope per docs/api-proxy.md §3:
        // `{ "error": { "message": ..., "type": "..." } }`
        assert_eq!(v["error"]["type"], "invalid_request_error");
        let message = v["error"]["message"].as_str().unwrap();
        assert!(
            message.contains("limit"),
            "413 message should reference the limit; got {message:?}"
        );
    }

    /// Audit HIGH-3 (#343): the body-limit middleware runs BEFORE
    /// the `/v1/messages` handler, so its rejection path must emit
    /// the Anthropic-shape envelope rather than the OpenAI-shape
    /// envelope — otherwise the Claude SDK's strict parser falls
    /// through to a generic exception. Same contract as the handler-
    /// side `into_anthropic_response()` for #336.
    #[tokio::test]
    async fn oversize_body_on_v1_messages_returns_anthropic_envelope_request_too_large() {
        let hub = Arc::new(Hub::new());
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], "http://unused");
        let app = build_router(build_state(snap, hub));

        let oversized = 2 * 1024 * 1024;
        let req = Request::builder()
            .method("POST")
            .uri("/v1/messages")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .header("content-length", oversized.to_string())
            .body(Body::from(
                r#"{"model":"my-gpt4","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}"#,
            ))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        // Anthropic-shape envelope per docs.anthropic.com/en/api/errors:
        // `{ "type": "error", "error": { "type": "...", "message": "..." } }`.
        assert_eq!(v["type"], "error", "Anthropic top-level discriminator");
        assert_eq!(
            v["error"]["type"], "request_too_large",
            "413 → Anthropic-canonical request_too_large per status-to-type mapping",
        );
        // OpenAI-only fields must be absent.
        assert!(v["error"].get("code").is_none());
        assert!(v["error"].get("param").is_none());
    }

    /// Audit MEDIUM-A (3rd audit) on #343: when the caller streams an
    /// oversize body without a declared Content-Length, the
    /// `enforce_request_body_limit` middleware skips its early reject
    /// (no length to compare), and the `Json<Value>` extractor's
    /// `DefaultBodyLimit` cap fires during read. That produces a
    /// `JsonRejection::BytesRejection`, which the handler MUST map
    /// to `RequestTooLarge` (413 + `error.type=="request_too_large"`)
    /// rather than `InvalidRequest` (400 + `"invalid_request_error"`)
    /// — the Claude SDK branches on `error.type=="request_too_large"`
    /// to mark requests as "non-retriable cap exceeded"; folding it
    /// into 400 breaks the retry-policy signal.
    #[tokio::test]
    async fn streaming_oversize_body_on_v1_messages_returns_413_request_too_large() {
        let hub = Arc::new(Hub::new());
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], "http://unused");
        let app = build_router(build_state(snap, hub));

        // Build a streaming body that yields > 1 MiB chunk-by-chunk
        // with NO upstream-set Content-Length. The middleware can't
        // decide on size and will pass through; the per-extractor
        // `DefaultBodyLimit` cap (set to `request_body_limit_bytes`
        // in `build_router`) fires on the read, surfacing as
        // `JsonRejection::BytesRejection`.
        let chunk = vec![b'x'; 200 * 1024]; // 200 KiB per chunk
        let stream =
            futures::stream::iter((0..10).map(move |_| Ok::<_, std::io::Error>(chunk.clone())));
        let body = Body::from_stream(stream);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/messages")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            // Intentionally NO Content-Length.
            .body(body)
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(
            resp.status(),
            StatusCode::PAYLOAD_TOO_LARGE,
            "streaming-oversize must surface as 413 request_too_large, NOT 400 invalid_request_error",
        );
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["type"], "error");
        assert_eq!(
            v["error"]["type"], "request_too_large",
            "Anthropic-canonical 413 → request_too_large; mistakenly folding into invalid_request_error \
             breaks the Claude SDK's retry-policy branch",
        );
    }

    /// Audit LOW-A (3rd audit) on #343: the path-match guard in
    /// `enforce_request_body_limit` must accept both `/v1/messages`
    /// and `/v1/messages/` (trailing slash) — axum's path
    /// normalization routes both to the Anthropic handler, but a
    /// strict `==` check on the bare form would miss the trailing-
    /// slash variant. SDKs don't add the slash; non-SDK callers
    /// (curl, custom clients) might.
    #[tokio::test]
    async fn oversize_body_on_v1_messages_trailing_slash_still_anthropic_envelope() {
        let hub = Arc::new(Hub::new());
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], "http://unused");
        let app = build_router(build_state(snap, hub));

        let oversized = 2 * 1024 * 1024;
        let req = Request::builder()
            .method("POST")
            .uri("/v1/messages/")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .header("content-length", oversized.to_string())
            .body(Body::from(
                r#"{"model":"my-gpt4","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}"#,
            ))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(
            v["type"], "error",
            "trailing-slash /v1/messages/ must still emit Anthropic envelope",
        );
        assert_eq!(v["error"]["type"], "request_too_large");
    }

    /// Companion to the above: duplicate Content-Length on /v1/messages
    /// also emits Anthropic envelope. Smuggling-rejection path runs
    /// in the same middleware as the body-limit reject.
    #[tokio::test]
    async fn duplicate_content_length_on_v1_messages_returns_anthropic_envelope() {
        let hub = Arc::new(Hub::new());
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], "http://unused");
        let app = build_router(build_state(snap, hub));

        let body =
            r#"{"model":"my-gpt4","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}"#;
        let mut req = Request::builder()
            .method("POST")
            .uri("/v1/messages")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body))
            .unwrap();
        req.headers_mut().append(
            axum::http::header::CONTENT_LENGTH,
            axum::http::HeaderValue::from(body.len()),
        );
        req.headers_mut().append(
            axum::http::header::CONTENT_LENGTH,
            axum::http::HeaderValue::from(body.len() + 1),
        );
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["type"], "error");
        assert_eq!(v["error"]["type"], "invalid_request_error");
    }

    /// Issue #159 audit MEDIUM-3: duplicate `Content-Length` headers
    /// are a classic request-smuggling vector — a server that acts on
    /// the first value while a downstream peer acts on the second can
    /// be tricked into framing the body wrongly. Per RFC 9110 §8.6 a
    /// server SHOULD reject the request rather than disambiguate.
    #[tokio::test]
    async fn duplicate_content_length_headers_return_400_invalid_request() {
        let hub = Arc::new(Hub::new());
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], "http://unused");
        let app = build_router(build_state(snap, hub));

        let body = r#"{"model":"my-gpt4","messages":[{"role":"user","content":"hi"}]}"#;
        let mut req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body))
            .unwrap();
        // Inject TWO Content-Length headers (simulating a smuggling
        // attempt). axum's HeaderMap supports `append` for duplicate
        // names; the middleware must reject rather than read the
        // first value.
        req.headers_mut().append(
            axum::http::header::CONTENT_LENGTH,
            axum::http::HeaderValue::from(body.len()),
        );
        req.headers_mut().append(
            axum::http::header::CONTENT_LENGTH,
            axum::http::HeaderValue::from(body.len() + 1),
        );

        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "invalid_request_error");
        let message = v["error"]["message"].as_str().unwrap();
        assert!(
            message.contains("Content-Length"),
            "smuggling-rejection message should mention Content-Length; got {message:?}"
        );
    }

    fn build_state_with_limit(snapshot: AisixSnapshot, hub: Arc<Hub>, limit: usize) -> ProxyState {
        let handle = SnapshotHandle::new(snapshot);
        let cfg = ProxyConfig {
            addr: "127.0.0.1:0".into(),
            request_body_limit_bytes: limit,
            real_ip: Default::default(),
            request_id: Default::default(),
            url_rewrites: Vec::new(),
            tls: None,
            thread_per_core: None,
            workers: None,
        };
        ProxyState::new(handle, hub, &cfg).without_cache()
    }

    /// `request_body_limit_bytes: 0` (the default) disables the cap
    /// entirely. The load-bearing detail is axum's BUILT-IN 2 MiB
    /// `DefaultBodyLimit`: merely skipping our `max(limit)` layer would
    /// still reject bodies over 2 MiB with a stock rejection, so the
    /// router must install `DefaultBodyLimit::disable()`. A 2.5 MiB body
    /// — over axum's built-in cap — must reach the handler on both the
    /// declared-Content-Length path and the chunked path.
    #[tokio::test]
    async fn zero_limit_admits_bodies_over_axums_builtin_cap() {
        let hub = Arc::new(Hub::new());
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], "http://unused");
        let app = build_router(build_state_with_limit(snap, hub, 0));

        let filler = "x".repeat(2 * 1024 * 1024 + 512 * 1024); // 2.5 MiB
        let body =
            format!(r#"{{"model":"my-gpt4","messages":[{{"role":"user","content":"{filler}"}}]}}"#);

        // Declared Content-Length path (the middleware's early check).
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .header("content-length", body.len().to_string())
            .body(Body::from(body.clone()))
            .unwrap();
        let resp = run(app.clone(), req).await;
        assert_ne!(
            resp.status(),
            StatusCode::PAYLOAD_TOO_LARGE,
            "limit 0 must not reject on declared size"
        );
        // The body parsed and dispatch ran (and failed on the unusable
        // upstream) — proving the request got PAST the extractor.
        assert!(
            resp.status().is_server_error(),
            "expected an upstream dispatch failure, got {}",
            resp.status()
        );

        // Chunked path (no Content-Length): this is the one axum's
        // built-in 2 MiB default would kill without `disable()`.
        let chunks: Vec<_> = body
            .into_bytes()
            .chunks(200 * 1024)
            .map(|c| c.to_vec())
            .collect();
        let stream = futures::stream::iter(chunks.into_iter().map(Ok::<_, std::io::Error>));
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from_stream(stream))
            .unwrap();
        let resp = run(app, req).await;
        assert_ne!(
            resp.status(),
            StatusCode::PAYLOAD_TOO_LARGE,
            "limit 0 must not cap a chunked body at axum's 2 MiB default"
        );
        assert!(resp.status().is_server_error());
    }

    /// A caller that sends everything it declared leaves the connection
    /// clean, so it can actually read the 413 the drain exists to
    /// deliver. This is the only outcome that is not a warning.
    #[tokio::test]
    async fn drain_reports_a_body_the_client_finished_sending() {
        let drain = drain_body(Body::from(vec![b'x'; 4096])).await;
        assert_eq!(drain.outcome, DrainOutcome::Completed);
        assert_eq!(drain.bytes, 4096);
    }

    /// Past the byte cap the rest of the body is never read, so the
    /// caller is likely to see a reset instead of the 413 — an operator
    /// has to be able to tell that apart from a clean drain.
    #[tokio::test]
    async fn drain_reports_hitting_the_byte_cap() {
        // One MiB over the 32 MiB cap, in chunks so nothing holds the
        // whole body at once.
        let chunk = vec![b'x'; 1024 * 1024];
        let stream =
            futures::stream::iter((0..33).map(move |_| Ok::<_, std::io::Error>(chunk.clone())));
        let drain = drain_body(Body::from_stream(stream)).await;
        assert_eq!(drain.outcome, DrainOutcome::CapReached);
        assert_eq!(drain.bytes, 32 * 1024 * 1024);
    }

    /// A client that dribbles (or stops) holds the drain until the time
    /// budget expires. `start_paused` advances the clock as soon as the
    /// runtime idles, so this costs no wall-clock time.
    #[tokio::test(start_paused = true)]
    async fn drain_reports_the_time_budget_expiring() {
        let head = futures::stream::iter([Ok::<_, std::io::Error>(vec![b'x'; 512])]);
        let never_ends = head.chain(futures::stream::pending());
        let drain = drain_body(Body::from_stream(never_ends)).await;
        assert_eq!(drain.outcome, DrainOutcome::Timeout);
        // What had been absorbed when the budget ran out, not zero.
        assert_eq!(drain.bytes, 512);
    }

    /// The client going away mid-body used to be folded into "drained
    /// fine" by a `while let Some(Ok(_))` loop, which is exactly the case
    /// an operator is trying to explain when a caller reports a reset.
    #[tokio::test]
    async fn drain_reports_the_client_vanishing_mid_body() {
        let stream = futures::stream::iter([
            Ok(vec![b'x'; 256]),
            Err(std::io::Error::other("connection reset")),
        ]);
        let drain = drain_body(Body::from_stream(stream)).await;
        assert_eq!(drain.outcome, DrainOutcome::ClientReadError);
        assert_eq!(drain.bytes, 256);
    }

    /// The rate limiter must never swallow the FIRST warning of a kind —
    /// that is the one an operator needs — and must not let one noisy
    /// outcome mask another's first occurrence.
    #[test]
    fn the_first_warning_of_each_abnormal_outcome_survives_the_rate_limit() {
        assert!(warn_allowed(DrainOutcome::Timeout));
        assert!(!warn_allowed(DrainOutcome::Timeout), "second within 1s");
        assert!(
            warn_allowed(DrainOutcome::ClientReadError),
            "a different outcome keeps its own budget"
        );
        assert!(
            !warn_allowed(DrainOutcome::Completed),
            "a clean drain is not a warning"
        );
    }

    /// A body-cap rejection short-circuits BEFORE any handler runs, and
    /// both the access log and the request metrics are emitted BY the
    /// handlers — so pre-fix a caller got a 413 the gateway kept no
    /// record of: nothing in the log, no `aisix_requests_total` sample.
    /// "Client reports 413, gateway shows nothing" was indistinguishable
    /// from the request never arriving.
    ///
    /// The metric is what this asserts, because it is per-`ProxyState`
    /// and so unaffected by whatever else the suite is doing; the log
    /// line — process-global tracing state, not safely assertable from a
    /// parallel unit test — is pinned end-to-end in the `body-edges` E2E.
    #[tokio::test]
    async fn body_cap_short_circuit_is_counted_like_any_other_terminal_path() {
        let hub = Arc::new(Hub::new());
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], "http://unused");
        let state = build_state(snap, hub); // 1 MiB cap from cfg()
        let app = build_router(state.clone());

        let oversized = 2 * 1024 * 1024;
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .header("content-length", oversized.to_string())
            .body(Body::from(
                r#"{"model":"my-gpt4","messages":[{"role":"user","content":"hi"}]}"#,
            ))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);

        let scrape = state.metrics.render();
        assert!(
            scrape.contains("aisix_requests_total") && scrape.contains(r#"status="413""#),
            "the 413 must be counted, got: {scrape}"
        );
    }

    /// The refusal is counted with the outcome of its drain, and both
    /// labels stay bounded: `endpoint` collapses to a route template even
    /// when the caller invents the path. This surface answers before
    /// authentication, so an unbounded label here would let anyone mint
    /// series at will (#451).
    #[tokio::test]
    async fn body_cap_rejection_is_counted_with_a_bounded_endpoint_and_its_outcome() {
        let hub = Arc::new(Hub::new());
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], "http://unused");
        let state = build_state(snap, hub); // 1 MiB cap from cfg()
        let app = build_router(state.clone());

        let req = Request::builder()
            .method("POST")
            .uri("/passthrough/openai/caller-invented-path-9d3f")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .header("content-length", (2 * 1024 * 1024).to_string())
            .body(Body::from("{}"))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);

        let scrape = state.metrics.render();
        let sample = scrape
            .lines()
            .find(|l| l.starts_with("aisix_proxy_request_body_limit_rejections_total{"))
            .unwrap_or_else(|| panic!("the refusal must be counted, got: {scrape}"));
        assert!(
            sample.contains(r#"endpoint="/passthrough/:provider/*rest""#),
            "the caller's path must not reach the label: {sample}"
        );
        assert!(sample.contains(r#"outcome="completed""#), "{sample}");
        assert!(sample.contains(r#"inbound_protocol="openai""#), "{sample}");
    }

    /// The chunked path reaches the handler, which rejects at its body
    /// extractor and returns before the dispatch tail — silent for the
    /// same reason, one layer further in. Locks the handler-side half.
    #[tokio::test]
    async fn chunked_oversize_rejection_is_counted_like_any_other_terminal_path() {
        let hub = Arc::new(Hub::new());
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], "http://unused");
        let state = build_state(snap, hub);
        let app = build_router(state.clone());

        let chunk = vec![b'x'; 200 * 1024];
        let stream =
            futures::stream::iter((0..10).map(move |_| Ok::<_, std::io::Error>(chunk.clone())));
        let req = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from_stream(stream))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);

        let scrape = state.metrics.render();
        assert!(
            scrape.contains(r#"status="413""#),
            "the 413 must be counted, got: {scrape}"
        );
    }

    /// The duplicate-Content-Length rejection is smuggling hygiene, not
    /// a size limit — it must keep firing when the cap is disabled.
    #[tokio::test]
    async fn zero_limit_still_rejects_duplicate_content_length() {
        let hub = Arc::new(Hub::new());
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], "http://unused");
        let app = build_router(build_state_with_limit(snap, hub, 0));

        let body = r#"{"model":"my-gpt4","messages":[{"role":"user","content":"hi"}]}"#;
        let mut req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body))
            .unwrap();
        req.headers_mut().append(
            axum::http::header::CONTENT_LENGTH,
            axum::http::HeaderValue::from(body.len()),
        );
        req.headers_mut().append(
            axum::http::header::CONTENT_LENGTH,
            axum::http::HeaderValue::from(body.len() + 1),
        );
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    /// Chunked oversize on a handler that used to take a bare
    /// `Json<Value>` extractor: the rejection must be the OpenAI
    /// envelope, not axum's stock `text/plain` 413. (The
    /// Content-Length path was already correct via the middleware;
    /// the chunked path leaked the stock rejection.)
    #[tokio::test]
    async fn chunked_oversize_on_v1_completions_returns_openai_envelope() {
        let hub = Arc::new(Hub::new());
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], "http://unused");
        let app = build_router(build_state(snap, hub));

        let chunk = vec![b'x'; 200 * 1024];
        let stream =
            futures::stream::iter((0..10).map(move |_| Ok::<_, std::io::Error>(chunk.clone())));
        let req = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from_stream(stream))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes)
            .expect("413 must carry the JSON envelope, not axum's text/plain rejection");
        assert_eq!(v["error"]["type"], "invalid_request_error");
    }

    /// Same contract for the raw-`Bytes` handlers (batches /
    /// fine-tuning).
    #[tokio::test]
    async fn chunked_oversize_on_v1_batches_returns_openai_envelope() {
        let hub = Arc::new(Hub::new());
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], "http://unused");
        let app = build_router(build_state(snap, hub));

        let chunk = vec![b'x'; 200 * 1024];
        let stream =
            futures::stream::iter((0..10).map(move |_| Ok::<_, std::io::Error>(chunk.clone())));
        let req = Request::builder()
            .method("POST")
            .uri("/v1/batches")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from_stream(stream))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes)
            .expect("413 must carry the JSON envelope, not axum's text/plain rejection");
        assert_eq!(v["error"]["type"], "invalid_request_error");
    }

    /// Chunked oversize multipart upload: axum's `MultipartError`
    /// classifies the cap hit as 413, and the handlers must preserve
    /// that instead of folding every multipart error into 400.
    #[tokio::test]
    async fn chunked_oversize_multipart_returns_413_envelope() {
        let hub = Arc::new(Hub::new());
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], "http://unused");
        let app = build_router(build_state(snap, hub));

        let boundary = "aisix-test-boundary-413";
        let mut body = Vec::new();
        body.extend_from_slice(
            format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; \
                 filename=\"a.wav\"\r\nContent-Type: audio/wav\r\n\r\n"
            )
            .as_bytes(),
        );
        body.extend(vec![b'x'; 2 * 1024 * 1024]); // over the 1 MiB test cap
        body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
        let chunks: Vec<_> = body.chunks(200 * 1024).map(|c| c.to_vec()).collect();
        let stream = futures::stream::iter(chunks.into_iter().map(Ok::<_, std::io::Error>));

        let req = Request::builder()
            .method("POST")
            .uri("/v1/audio/transcriptions")
            .header("authorization", "Bearer sk-caller")
            .header(
                "content-type",
                format!("multipart/form-data; boundary={boundary}"),
            )
            .body(Body::from_stream(stream))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "invalid_request_error");
        let message = v["error"]["message"].as_str().unwrap();
        assert!(
            message.contains("limit"),
            "413 message should reference the limit; got {message:?}"
        );
    }

    /// The passthrough tunnel reads its body manually (`to_bytes`), so
    /// the `0` sentinel has to be widened there too — this is the site
    /// the first audit round caught unconverted, where every POST got
    /// `413 request body exceeds 0-byte limit` on the new default.
    #[tokio::test]
    async fn zero_limit_passthrough_post_is_not_rejected() {
        let hub = Arc::new(Hub::new());
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], "http://unused");
        let app = build_router(build_state_with_limit(snap, hub, 0));

        let req = Request::builder()
            .method("POST")
            .uri("/passthrough/openai/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"model":"gpt-4o","input":"hi"}"#))
            .unwrap();
        let resp = run(app, req).await;
        assert_ne!(
            resp.status(),
            StatusCode::PAYLOAD_TOO_LARGE,
            "limit 0 must not reject the passthrough body"
        );
        // Past the body read; the dispatch then failed on the unusable
        // upstream.
        assert!(
            resp.status().is_server_error(),
            "expected an upstream dispatch failure, got {}",
            resp.status()
        );
    }

    /// With a configured cap, a chunked over-limit passthrough body is
    /// a 413 in the envelope — and a transport fault stays a 400, no
    /// longer mislabelled as `RequestTooLarge`.
    #[tokio::test]
    async fn chunked_oversize_on_passthrough_returns_openai_envelope() {
        let hub = Arc::new(Hub::new());
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], "http://unused");
        let app = build_router(build_state(snap, hub));

        let chunk = vec![b'x'; 200 * 1024];
        let stream =
            futures::stream::iter((0..10).map(move |_| Ok::<_, std::io::Error>(chunk.clone())));
        let req = Request::builder()
            .method("POST")
            .uri("/passthrough/openai/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from_stream(stream))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value =
            serde_json::from_slice(&bytes).expect("413 must carry the JSON envelope");
        assert_eq!(v["error"]["type"], "invalid_request_error");
        let message = v["error"]["message"].as_str().unwrap();
        assert!(
            message.contains("limit"),
            "413 message should reference the limit; got {message:?}"
        );
    }

    /// Issue #159 companion: a body within the cap must NOT be
    /// rejected — the middleware short-circuits ONLY when the
    /// Content-Length exceeds the cap, leaving normal traffic
    /// untouched. Without this guard, a regression that always-
    /// rejected (e.g. comparing the wrong field) would be invisible
    /// since most existing tests don't set Content-Length.
    #[tokio::test]
    async fn within_limit_body_is_not_rejected_by_middleware() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-ok",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            })))
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        let app = build_router(build_state(snap, hub));

        let body = r#"{"model":"my-gpt4","messages":[{"role":"user","content":"hi"}]}"#;
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .header("content-length", body.len().to_string())
            .body(Body::from(body))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn upstream_429_passes_through_with_openai_envelope() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        let app = build_router(build_state(snap, hub));

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"my-gpt4","messages":[{"role":"user","content":"hi"}]}"#,
            ))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "upstream_error");
    }

    /// Issues #322 + #327: when an OpenAI upstream returns a coded
    /// 4xx with the standard `{error:{message,type,code,param}}`
    /// envelope, the gateway:
    /// - preserves `message`, `code`, and `param` verbatim so SDK
    ///   retry logic that branches on `error.code` keeps working;
    /// - normalises `error.type` to the DP-stable token
    ///   `"upstream_error"`, hiding the upstream's private taxonomy
    ///   from the customer (the upstream `type` here —
    ///   `"upstream_test_fixture"` — is mock-llm's internal label
    ///   and must not bleed through).
    #[tokio::test]
    async fn upstream_openai_4xx_forwards_code_and_param_but_normalises_type_per_issue_327() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(429).set_body_raw(
                br#"{"error":{"message":"upstream forced 429","type":"upstream_test_fixture","code":"forced_429","param":"model"}}"#.as_slice(),
                "application/json",
            ))
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        let app = build_router(build_state(snap, hub));

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"my-gpt4","messages":[{"role":"user","content":"hi"}]}"#,
            ))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
        let bytes = to_bytes(resp.into_body(), 2048).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["message"], "upstream forced 429");
        // Per #327: `error.type` is the DP-stable taxonomy, NOT the
        // upstream's `type`. The upstream's `upstream_test_fixture`
        // token must NOT leak to the customer envelope.
        assert_eq!(v["error"]["type"], "upstream_error");
        // Per #322: `error.code` and `error.param` ARE preserved so
        // SDK retry logic can branch on the granular code.
        assert_eq!(v["error"]["code"], "forced_429");
        assert_eq!(v["error"]["param"], "model");
    }

    /// Issue #322 fallback contract: when the upstream body is not a
    /// recognisable JSON envelope (HTML error page, garbled text), the
    /// gateway must NOT crash or surface raw bytes; it falls back to
    /// the generic `upstream_error` envelope with the truncated body
    /// as `message`. This pins the content-type guard.
    #[tokio::test]
    async fn upstream_4xx_non_json_body_falls_back_to_generic_envelope() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(400).set_body_raw(
                b"<html><body>403 Forbidden by WAF</body></html>".as_slice(),
                "text/html",
            ))
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        let app = build_router(build_state(snap, hub));

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"my-gpt4","messages":[{"role":"user","content":"hi"}]}"#,
            ))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let bytes = to_bytes(resp.into_body(), 2048).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        // Generic envelope: type=upstream_error, message contains the
        // truncated raw body (no JSON parse attempted on text/html).
        assert_eq!(v["error"]["type"], "upstream_error");
        assert!(v["error"].get("code").is_none() || v["error"]["code"].is_null());
    }

    /// Issue #322 sanity check on the 5xx branch: upstream 5xx still
    /// collapses to 502 with the generic envelope AND the upstream
    /// `error.message` is suppressed. Engine names / shard ids / queue
    /// depth routinely appear in upstream 5xx bodies (in this fixture:
    /// "engine offline shard 47") — those are operator-internal and
    /// must not bleed through to the customer envelope.
    #[tokio::test]
    async fn upstream_openai_5xx_with_json_envelope_collapses_and_redacts_message() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(503).set_body_raw(
                br#"{"error":{"message":"engine offline shard 47","type":"server_error","code":"engine_overloaded"}}"#.as_slice(),
                "application/json",
            ))
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        let app = build_router(build_state(snap, hub));

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"my-gpt4","messages":[{"role":"user","content":"hi"}]}"#,
            ))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
        let bytes = to_bytes(resp.into_body(), 2048).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "upstream_error");
        let msg = v["error"]["message"].as_str().unwrap();
        assert!(
            !msg.contains("engine offline") && !msg.contains("shard 47"),
            "upstream 5xx `error.message` must NOT leak to customer; got: {msg:?}"
        );
        // Upstream `code` (engine_overloaded) must also not pass
        // through on 5xx.
        assert!(
            v["error"].get("code").is_none() || v["error"]["code"].is_null(),
            "upstream 5xx `error.code` must not pass through; got code={:?}",
            v["error"]["code"]
        );
    }

    /// Cross-provider contract: Anthropic upstream 5xx → client sees an
    /// OpenAI-shape envelope `{error:{type:"upstream_error",...}}` with
    /// status 502 (collapsed per `BridgeError::http_status`, see
    /// crates/aisix-gateway/src/bridge.rs).
    #[tokio::test]
    async fn upstream_anthropic_5xx_collapses_to_502_with_openai_envelope() {
        use aisix_provider_anthropic::AnthropicBridge;

        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(503).set_body_string(
                r#"{"type":"error","error":{"type":"overloaded_error","message":"upstream busy"}}"#,
            ))
            .mount(&upstream)
            .await;

        let snap = AisixSnapshot::new();
        snap.provider_keys
            .insert(matrix_anthropic_pk(&upstream.uri()));
        snap.models.insert(anthropic_model_entry("my-claude"));
        snap.apikeys
            .insert(apikey_entry("sk-caller", &["my-claude"]));
        let hub = Arc::new(Hub::new());
        hub.register_specialized("anthropic", Arc::new(AnthropicBridge::new()));
        let app = build_router(build_state(snap, hub));

        let body = serde_json::json!({
            "model": "my-claude",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
        let v: serde_json::Value =
            serde_json::from_slice(&to_bytes(resp.into_body(), 1024).await.unwrap()).unwrap();
        // Anthropic-shape leaks must not bleed through to the client.
        assert_eq!(v["error"]["type"], "upstream_error");
        assert!(v["error"]["message"].is_string());
    }

    /// Cross-provider 4xx forwarding (issue #327): Anthropic upstream
    /// 400 reaches the OpenAI-client side with `error.type` normalised
    /// to the DP-stable `"upstream_error"` token — Anthropic's private
    /// taxonomy (`invalid_request_error`, `authentication_error`, etc.)
    /// must not bleed through.
    #[tokio::test]
    async fn upstream_anthropic_400_normalises_type_to_upstream_error() {
        use aisix_provider_anthropic::AnthropicBridge;

        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(400).set_body_raw(
                br#"{"type":"error","error":{"type":"invalid_request_error","message":"bad input"}}"#.as_slice(),
                "application/json",
            ))
            .mount(&upstream)
            .await;

        let snap = AisixSnapshot::new();
        snap.provider_keys
            .insert(matrix_anthropic_pk(&upstream.uri()));
        snap.models.insert(anthropic_model_entry("my-claude"));
        snap.apikeys
            .insert(apikey_entry("sk-caller", &["my-claude"]));
        let hub = Arc::new(Hub::new());
        hub.register_specialized("anthropic", Arc::new(AnthropicBridge::new()));
        let app = build_router(build_state(snap, hub));

        let body = serde_json::json!({
            "model": "my-claude",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let v: serde_json::Value =
            serde_json::from_slice(&to_bytes(resp.into_body(), 1024).await.unwrap()).unwrap();
        assert_eq!(v["error"]["type"], "upstream_error");
        assert_eq!(v["error"]["message"], "bad input");
        // Anthropic `invalid_request_error` doesn't derive an OpenAI
        // string code — translation table emits `code: null`.
        assert!(v["error"].get("code").is_none() || v["error"]["code"].is_null());
    }

    /// Issue #322 + #327 cross-wire contract: Anthropic upstream
    /// `rate_limit_error` must derive OpenAI `error.code =
    /// rate_limit_exceeded` (so SDK retry logic that switches on
    /// `error.code` recognises the rate-limit failure regardless of
    /// upstream), while `error.type` stays as the DP-stable
    /// `"upstream_error"` (per #327, Anthropic's `rate_limit_error`
    /// token must not bleed through).
    #[tokio::test]
    async fn upstream_anthropic_rate_limit_derives_openai_rate_limit_exceeded_code() {
        use aisix_provider_anthropic::AnthropicBridge;

        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(429).set_body_raw(
                br#"{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}"#.as_slice(),
                "application/json",
            ))
            .mount(&upstream)
            .await;

        let snap = AisixSnapshot::new();
        snap.provider_keys
            .insert(matrix_anthropic_pk(&upstream.uri()));
        snap.models.insert(anthropic_model_entry("my-claude"));
        snap.apikeys
            .insert(apikey_entry("sk-caller", &["my-claude"]));
        let hub = Arc::new(Hub::new());
        hub.register_specialized("anthropic", Arc::new(AnthropicBridge::new()));
        let app = build_router(build_state(snap, hub));

        let body = serde_json::json!({
            "model": "my-claude",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
        let v: serde_json::Value =
            serde_json::from_slice(&to_bytes(resp.into_body(), 1024).await.unwrap()).unwrap();
        // Per #327: `error.type` is the DP-stable token, never the
        // upstream's. Per #322: `error.code` is the derived OpenAI
        // string code so SDK retry logic fires correctly.
        assert_eq!(v["error"]["type"], "upstream_error");
        assert_eq!(v["error"]["code"], "rate_limit_exceeded");
        assert_eq!(v["error"]["message"], "slow down");
    }

    /// Garbage upstream body (200 + non-JSON) must surface as 502 with
    /// `error.type = "upstream_decode_error"` — distinct from the 4xx/5xx
    /// `upstream_error` token so dashboards can tell parsing failures
    /// apart from upstream errors.
    #[tokio::test]
    async fn upstream_unparseable_body_returns_502_decode_error_envelope() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_string("{not valid json"))
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        let app = build_router(build_state(snap, hub));

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"my-gpt4","messages":[{"role":"user","content":"hi"}]}"#,
            ))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
        let v: serde_json::Value =
            serde_json::from_slice(&to_bytes(resp.into_body(), 1024).await.unwrap()).unwrap();
        assert_eq!(v["error"]["type"], "upstream_decode_error");
    }

    #[tokio::test]
    async fn provider_without_registered_bridge_returns_503() {
        // Snapshot has a Model targeting openai, but the Hub is empty.
        let hub = Arc::new(Hub::new());
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], "http://unused");
        let app = build_router(build_state(snap, hub));

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"my-gpt4","messages":[{"role":"user","content":"hi"}]}"#,
            ))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn streaming_response_emits_sse_then_done_sentinel() {
        let upstream = MockServer::start().await;
        let sse = "\
data: {\"id\":\"up-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"up-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"up-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
data: [DONE]\n\n";
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse),
            )
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        let app = build_router(build_state(snap, hub));

        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default()
            .contains("text/event-stream"));

        // Drain the body, decode SSE events, assert we got at least one
        // delta chunk plus the terminating [DONE].
        let mut body_stream = resp.into_body().into_data_stream();
        let mut decoder = SseDecoder::new();
        let mut events = Vec::new();
        while let Some(chunk) = body_stream.next().await {
            let bytes = chunk.unwrap();
            events.extend(decoder.feed(bytes.as_ref()));
        }
        assert!(events.contains(&SseEvent::Done), "missing [DONE] sentinel");
        let data_count = events
            .iter()
            .filter(|e| matches!(e, SseEvent::Data(_)))
            .count();
        assert!(
            data_count >= 2,
            "expected at least two chat chunks, got {data_count}"
        );
    }

    /// Issue #177: per docs/api-proxy.md §5, abnormal upstream
    /// stream termination must close the response WITHOUT `[DONE]`
    /// — SDK consumers that key off `[DONE]` for clean-completion
    /// signal need to detect truncation. The previous behavior
    /// emitted `[DONE]` after the SSE error event, masking
    /// truncated responses as complete.
    #[tokio::test]
    async fn streaming_response_omits_done_when_upstream_returns_invalid_json_mid_stream() {
        let upstream = MockServer::start().await;
        // Upstream emits two valid SSE chunks then a malformed JSON
        // payload. The malformed payload triggers `serde_json::from_str`
        // to fail in the bridge's `build_chunk_stream`, surfacing as
        // `BridgeError::UpstreamDecode` to `build_sse_stream`, which
        // emits an SSE `event: error` frame. After that frame the
        // proxy MUST NOT emit `[DONE]`.
        let sse = "\
data: {\"id\":\"up-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"up-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial \"},\"finish_reason\":null}]}\n\n\
data: <not valid json>\n\n";
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse),
            )
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        let app = build_router(build_state(snap, hub));

        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);

        // Drain the wire bytes — at this layer we want byte-level
        // assertions, not just decoded SSE events, because the
        // contract being verified is "did `[DONE]` appear at all
        // on the wire".
        let mut body_stream = resp.into_body().into_data_stream();
        let mut wire = Vec::new();
        while let Some(chunk) = body_stream.next().await {
            wire.extend_from_slice(chunk.unwrap().as_ref());
        }
        let wire_str = String::from_utf8(wire).expect("SSE bytes are utf8");

        // Per docs §5: NO `[DONE]` after abnormal termination.
        assert!(
            !wire_str.contains("data: [DONE]"),
            "abnormal termination MUST close without [DONE]; got wire:\n{wire_str}"
        );
        // The error event MUST be emitted so SDK consumers see a
        // failure signal.
        assert!(
            wire_str.contains("event: error"),
            "abnormal termination MUST emit `event: error`; got wire:\n{wire_str}"
        );
        // The error payload MUST be valid OpenAI-envelope JSON
        // (the SDK does `JSON.parse(sse.data)` BEFORE checking
        // event type, so plain-string payloads yield a SyntaxError
        // instead of the typed APIError callers expect).
        let err_event_idx = wire_str.find("event: error\n").unwrap();
        let after_err = &wire_str[err_event_idx + "event: error\n".len()..];
        let data_line = after_err
            .lines()
            .find(|l| l.starts_with("data: "))
            .expect("error event followed by a data line");
        let json_payload = &data_line["data: ".len()..];
        let parsed: serde_json::Value = serde_json::from_str(json_payload)
            .expect("error frame data must be valid OpenAI-envelope JSON");
        assert!(
            parsed.get("error").is_some(),
            "error frame data must be `{{\"error\": {{...}}}}` shape; got {json_payload}"
        );
    }

    /// Issue #204: streaming responses MUST run output guardrails — pre-fix
    /// the streaming path skipped them, so a `kind: "keyword"` deny-list was
    /// bypassable by setting `stream: true`.
    ///
    /// Issue #466: output guardrails must also HOLD content back while
    /// streaming. `keyword` now inherits the default hold-back policy
    /// ([`StreamOutputPolicy::BufferFull`], fail-closed), so a blocked
    /// streaming response NEVER puts the forbidden content on the wire.
    /// This test pins:
    ///
    ///   - 200 OK + SSE wire shape (the request itself is well-formed)
    ///   - upstream IS hit (output guardrails run AFTER the upstream call)
    ///   - an SSE `event: error` frame with the OpenAI `content_filter` envelope
    ///   - NO terminal `[DONE]` (a guardrail block is an abnormal termination)
    ///   - the matched literal "secret-string" does NOT appear ANYWHERE on
    ///     the wire — the hold-back means the offending content is never
    ///     emitted (the #466 fix; previously the live-forwarded chunks
    ///     leaked it before the end-of-stream check).
    #[tokio::test]
    async fn streaming_output_guardrail_blocks_with_sse_error_event_and_no_done() {
        let upstream = MockServer::start().await;
        // Upstream emits 3 SSE chunks: role, then content containing
        // the forbidden literal, then the terminal stop. The full
        // assistant content the guardrail evaluates is "leak: secret-string"
        // which the keyword guardrail at "secret-string" must block.
        let sse = "\
data: {\"id\":\"up-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"up-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"leak: secret-string\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"up-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
data: [DONE]\n\n";
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse),
            )
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        let state = build_state(snap, hub);
        seed_guardrail(
            &state.snapshot,
            "g-stream-output",
            r#"{"name":"stream-output-guard","kind":"keyword","hook_point":"output","patterns":[{"kind":"literal","value":"secret-string"}]}"#,
        );
        let app = build_router(state);

        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);

        let mut body_stream = resp.into_body().into_data_stream();
        let mut wire = Vec::new();
        while let Some(chunk) = body_stream.next().await {
            wire.extend_from_slice(chunk.unwrap().as_ref());
        }
        let wire_str = String::from_utf8(wire).expect("SSE bytes are utf8");

        // #466: the default hold-back policy means the forbidden content is
        // never forwarded — the matched literal appears NOWHERE on the wire
        // (pre-fix the live-forwarded chunks leaked it before the check).
        assert!(
            !wire_str.contains("secret-string"),
            "hold-back guardrail leaked the matched content onto the wire; got:\n{wire_str}"
        );

        // Per docs §5 abnormal-termination contract (the guardrail
        // block is the streaming-equivalent of an abnormal close):
        // NO `[DONE]` after the error event. SDK consumers that key
        // off `[DONE]` need to detect the truncation.
        assert!(
            !wire_str.contains("data: [DONE]"),
            "blocked stream MUST close without [DONE]; got wire:\n{wire_str}"
        );
        // SSE `event: error` frame MUST appear so SDK consumers see
        // a failure signal.
        assert!(
            wire_str.contains("event: error"),
            "blocked stream MUST emit `event: error`; got wire:\n{wire_str}"
        );
        // The error frame's data MUST be valid OpenAI-envelope JSON
        // with `error.type: "content_filter"` (parallel to #153's
        // non-streaming contract).
        let err_event_idx = wire_str.find("event: error\n").unwrap();
        let after_err = &wire_str[err_event_idx + "event: error\n".len()..];
        let data_line = after_err
            .lines()
            .find(|l| l.starts_with("data: "))
            .expect("error event followed by a data line");
        let json_payload = &data_line["data: ".len()..];
        let parsed: serde_json::Value = serde_json::from_str(json_payload)
            .expect("error frame data must be valid OpenAI-envelope JSON");
        assert_eq!(
            parsed["error"]["type"], "content_filter",
            "error.type must mark the guardrail block; got {json_payload}"
        );
        // Per #153: the matched literal MUST NOT appear inside the
        // error frame envelope (the *error*, not the pre-emitted
        // chunks — those carry the partial content that buffer-
        // then-check accepts as a known trade-off).
        let error_message = parsed["error"]["message"].as_str().unwrap();
        assert!(
            !error_message.contains("secret-string"),
            "guardrail leaked the matched literal in the error envelope; got {error_message:?}"
        );
        assert_eq!(
            error_message, "response blocked by content policy (guardrail 'stream-output-guard')",
            "wire-level message stays redacted per #153 but names the guardrail per #519 B.4b"
        );
    }

    /// #448 parity (streaming): the chat streaming output guardrail buffered
    /// only `delta.content`, so a blocked literal in a tool-call's `arguments`
    /// leaked (chat non-streaming + /v1/messages streaming already scan tool
    /// calls). With the fix, tool-call name + arguments are scanned at
    /// end-of-stream; a blocked literal blocks the stream (error frame) and is
    /// held back (never on the wire) under the default BufferFull policy.
    #[tokio::test]
    async fn streaming_output_guardrail_blocks_tool_call_arguments() {
        let upstream = MockServer::start().await;
        let c1 = serde_json::json!({"id":"up-1","model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]});
        // Tool-call delta carrying the forbidden literal in `arguments` — the
        // pre-fix path never scanned this.
        let c2 = serde_json::json!({"id":"up-1","model":"gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"q\":\"secret-string\"}"}}]},"finish_reason":null}]});
        let c3 = serde_json::json!({"id":"up-1","model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]});
        let sse = format!("data: {c1}\n\ndata: {c2}\n\ndata: {c3}\n\ndata: [DONE]\n\n");
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse),
            )
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        let state = build_state(snap, hub);
        seed_guardrail(
            &state.snapshot,
            "g-stream-tc",
            r#"{"name":"stream-tc-guard","kind":"keyword","hook_point":"output","patterns":[{"kind":"literal","value":"secret-string"}]}"#,
        );
        let app = build_router(state);

        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);

        let mut body_stream = resp.into_body().into_data_stream();
        let mut wire = Vec::new();
        while let Some(chunk) = body_stream.next().await {
            wire.extend_from_slice(chunk.unwrap().as_ref());
        }
        let wire_str = String::from_utf8(wire).expect("SSE bytes are utf8");

        assert!(
            !wire_str.contains("secret-string"),
            "tool-call arguments leaked despite output block; got:\n{wire_str}"
        );
        assert!(
            wire_str.contains("event: error"),
            "blocked stream must emit `event: error`; got:\n{wire_str}"
        );
        let idx = wire_str
            .find("event: error\n")
            .expect("error event present");
        let data_line = wire_str[idx..]
            .lines()
            .find(|l| l.starts_with("data: "))
            .expect("error event followed by a data line");
        let parsed: serde_json::Value =
            serde_json::from_str(&data_line["data: ".len()..]).expect("valid error envelope");
        assert_eq!(parsed["error"]["type"], "content_filter");
    }

    /// P2 (#379): like the keyword guardrail above (which now holds back by
    /// default, #466), `azure_content_safety_text_moderation` keeps offending
    /// content off the wire — here via the configurable `Window` policy
    /// rather than the default `BufferFull`. A blocked streaming response
    /// NEVER puts the offending content on the wire.
    #[tokio::test]
    async fn streaming_text_moderation_blocks_and_holds_content_back_no_leak() {
        let upstream = MockServer::start().await;
        let sse = "\
data: {\"id\":\"up-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"up-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"this is harmful text\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"up-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
data: [DONE]\n\n";
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse),
            )
            .mount(&upstream)
            .await;

        // Azure Content Safety mock returns high severity → block.
        let acs = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/contentsafety/text:analyze"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "categoriesAnalysis": [{"category": "Hate", "severity": 6}],
                "blocklistsMatch": []
            })))
            .mount(&acs)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        let state = build_state(snap, hub);
        seed_guardrail(
            &state.snapshot,
            "g-textmod-output",
            &format!(
                r#"{{"name":"textmod","kind":"azure_content_safety_text_moderation","hook_point":"output","endpoint":"{}","api_key":"k"}}"#,
                acs.uri()
            ),
        );
        let app = build_router(state);

        let body = serde_json::json!({"model":"my-gpt4","messages":[{"role":"user","content":"hi"}],"stream":true});
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);

        let mut body_stream = resp.into_body().into_data_stream();
        let mut wire = Vec::new();
        while let Some(chunk) = body_stream.next().await {
            wire.extend_from_slice(chunk.unwrap().as_ref());
        }
        let wire_str = String::from_utf8(wire).expect("SSE bytes are utf8");

        // The hold-back guarantee: the harmful content NEVER reached the
        // wire (held in `pending`, dropped on block).
        assert!(
            !wire_str.contains("harmful text"),
            "hold-back must keep blocked content off the wire; got:\n{wire_str}"
        );
        assert!(
            wire_str.contains("event: error"),
            "blocked stream must emit `event: error`; got:\n{wire_str}"
        );
        assert!(
            !wire_str.contains("data: [DONE]"),
            "blocked stream must omit [DONE]; got:\n{wire_str}"
        );
        let idx = wire_str.find("event: error\n").unwrap();
        let data_line = wire_str[idx..]
            .lines()
            .find(|l| l.starts_with("data: "))
            .unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&data_line["data: ".len()..]).unwrap();
        assert_eq!(parsed["error"]["type"], "content_filter");
    }

    /// Clean content is held back, scanned, then released in full + [DONE].
    #[tokio::test]
    async fn streaming_text_moderation_releases_clean_content() {
        let upstream = MockServer::start().await;
        let sse = "\
data: {\"id\":\"up-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"perfectly fine answer\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"up-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
data: [DONE]\n\n";
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse),
            )
            .mount(&upstream)
            .await;
        let acs = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/contentsafety/text:analyze"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "categoriesAnalysis": [{"category": "Hate", "severity": 0}],
                "blocklistsMatch": []
            })))
            .mount(&acs)
            .await;
        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        let state = build_state(snap, hub);
        seed_guardrail(
            &state.snapshot,
            "g-textmod-clean",
            &format!(
                r#"{{"name":"textmod","kind":"azure_content_safety_text_moderation","hook_point":"output","endpoint":"{}","api_key":"k"}}"#,
                acs.uri()
            ),
        );
        let app = build_router(state);
        let body = serde_json::json!({"model":"my-gpt4","messages":[{"role":"user","content":"hi"}],"stream":true});
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let mut body_stream = resp.into_body().into_data_stream();
        let mut wire = Vec::new();
        while let Some(c) = body_stream.next().await {
            wire.extend_from_slice(c.unwrap().as_ref());
        }
        let wire_str = String::from_utf8(wire).expect("utf8");
        assert!(
            wire_str.contains("perfectly fine answer"),
            "clean content must be released after the scan; got:\n{wire_str}"
        );
        assert!(
            wire_str.contains("data: [DONE]"),
            "clean stream must end with [DONE]; got:\n{wire_str}"
        );
        assert!(
            !wire_str.contains("event: error"),
            "clean stream must not emit an error frame; got:\n{wire_str}"
        );
    }

    /// Drive a streaming chat through a seeded text-moderation guardrail
    /// and return the raw SSE wire bytes. `guardrail_cfg` is the full
    /// guardrail JSON (with the ACS mock endpoint already substituted).
    async fn run_textmod_stream(guardrail_cfg: &str, upstream_sse: &str) -> String {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(upstream_sse.to_owned()),
            )
            .mount(&upstream)
            .await;
        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        let state = build_state(snap, hub);
        seed_guardrail(&state.snapshot, "g-tm", guardrail_cfg);
        let app = build_router(state);
        let body = serde_json::json!({"model":"my-gpt4","messages":[{"role":"user","content":"hi"}],"stream":true});
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let mut body_stream = resp.into_body().into_data_stream();
        let mut wire = Vec::new();
        while let Some(c) = body_stream.next().await {
            wire.extend_from_slice(c.unwrap().as_ref());
        }
        String::from_utf8(wire).expect("utf8")
    }

    async fn acs_mock(severity: u8) -> MockServer {
        let acs = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/contentsafety/text:analyze"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "categoriesAnalysis": [{"category": "Hate", "severity": severity}],
                "blocklistsMatch": []
            })))
            .mount(&acs)
            .await;
        acs
    }

    fn two_content_chunks(a: &str, b: &str) -> String {
        format!(
            "data: {{\"id\":\"u\",\"model\":\"gpt-4o\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"{a}\"}},\"finish_reason\":null}}]}}\n\n\
data: {{\"id\":\"u\",\"model\":\"gpt-4o\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"{b}\"}},\"finish_reason\":null}}]}}\n\n\
data: {{\"id\":\"u\",\"model\":\"gpt-4o\",\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"stop\"}}]}}\n\n\
data: [DONE]\n\n"
        )
    }

    /// H1: Window mode blocks MID-STREAM (small window so the first window
    /// trips before end-of-stream) and leaks nothing.
    #[tokio::test]
    async fn streaming_text_moderation_window_blocks_mid_stream() {
        let acs = acs_mock(6).await;
        let cfg = format!(
            r#"{{"name":"tm","kind":"azure_content_safety_text_moderation","hook_point":"output","endpoint":"{}","api_key":"k","stream_processing_mode":"window","window_size":5,"window_overlap_size":1}}"#,
            acs.uri()
        );
        let wire = run_textmod_stream(&cfg, &two_content_chunks("hello ", "world!")).await;
        assert!(
            !wire.contains("hello"),
            "mid-stream block must not leak window content; got:\n{wire}"
        );
        assert!(
            !wire.contains("world"),
            "mid-stream block must not leak later content; got:\n{wire}"
        );
        assert!(
            wire.contains("event: error"),
            "expected content_filter frame; got:\n{wire}"
        );
        assert!(
            !wire.contains("data: [DONE]"),
            "blocked stream omits [DONE]; got:\n{wire}"
        );
    }

    /// H1: Window mode releases multiple clean windows (exercises the
    /// mid-stream flush + overlap retention), ending with [DONE].
    #[tokio::test]
    async fn streaming_text_moderation_window_releases_clean_multiwindow() {
        let acs = acs_mock(0).await;
        let cfg = format!(
            r#"{{"name":"tm","kind":"azure_content_safety_text_moderation","hook_point":"output","endpoint":"{}","api_key":"k","stream_processing_mode":"window","window_size":5,"window_overlap_size":2}}"#,
            acs.uri()
        );
        let wire = run_textmod_stream(&cfg, &two_content_chunks("hello ", "world!")).await;
        assert!(
            wire.contains("hello"),
            "clean windows must be released; got:\n{wire}"
        );
        assert!(
            wire.contains("world"),
            "all clean content must be released; got:\n{wire}"
        );
        assert!(
            wire.contains("data: [DONE]"),
            "clean stream ends with [DONE]; got:\n{wire}"
        );
        assert!(
            !wire.contains("event: error"),
            "clean stream emits no error; got:\n{wire}"
        );
    }

    /// H1: BufferFull cap exceeded with fail_closed → block, no leak.
    #[tokio::test]
    async fn streaming_text_moderation_buffer_full_cap_fail_closed_blocks() {
        let acs = acs_mock(0).await; // severity irrelevant — the cap trips first
        let cfg = format!(
            r#"{{"name":"tm","kind":"azure_content_safety_text_moderation","hook_point":"output","endpoint":"{}","api_key":"k","stream_processing_mode":"buffer_full","max_buffer_bytes":4,"on_buffer_exceeded":"fail_closed"}}"#,
            acs.uri()
        );
        let wire = run_textmod_stream(&cfg, &two_content_chunks("abcd", "efghij")).await;
        assert!(
            !wire.contains("abcd"),
            "fail-closed cap must not leak buffered content; got:\n{wire}"
        );
        assert!(
            wire.contains("event: error"),
            "cap fail-closed must emit content_filter; got:\n{wire}"
        );
        assert!(
            !wire.contains("data: [DONE]"),
            "cap-blocked stream omits [DONE]; got:\n{wire}"
        );
    }

    /// H1: BufferFull cap exceeded with fail_open → release held + forward
    /// the rest live, ending with [DONE].
    #[tokio::test]
    async fn streaming_text_moderation_buffer_full_cap_fail_open_releases() {
        let acs = acs_mock(0).await;
        let cfg = format!(
            r#"{{"name":"tm","kind":"azure_content_safety_text_moderation","hook_point":"output","endpoint":"{}","api_key":"k","stream_processing_mode":"buffer_full","max_buffer_bytes":4,"on_buffer_exceeded":"fail_open"}}"#,
            acs.uri()
        );
        let wire = run_textmod_stream(&cfg, &two_content_chunks("abcd", "efghij")).await;
        assert!(
            wire.contains("abcd"),
            "fail-open cap must release held content; got:\n{wire}"
        );
        assert!(
            wire.contains("data: [DONE]"),
            "fail-open released stream ends with [DONE]; got:\n{wire}"
        );
        assert!(
            !wire.contains("event: error"),
            "fail-open release emits no error; got:\n{wire}"
        );
    }

    // ---- regression coverage for issue #107 -------------------------
    // Pre-fix only /v1/chat/completions enforced rate-limit / budget;
    // every other LLM endpoint silently bypassed both. The test below
    // pins /v1/embeddings — representative of the class — to ensure
    // the gate fires after this PR. Adding the same coverage to every
    // endpoint would multiply the test surface without buying signal,
    // since the gate is centralised in `crate::quota::enforce`. If
    // any individual handler ever stops calling it, that handler's
    // own tests would still catch the breakage on the budget path
    // (BudgetExceeded surfaces as a 4xx the existing tests assert on).

    #[tokio::test]
    async fn rate_limit_rpm_applies_to_embeddings_endpoint_issue_107() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "object": "list",
                "model": "text-embedding-3-small",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
                "usage": {"prompt_tokens": 5, "total_tokens": 5}
            })))
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = seed_snapshot_with_limits(
            "my-gpt4",
            &["my-gpt4"],
            &upstream.uri(),
            serde_json::json!({"rpm": 1}),
        );
        let state = build_state(snap, hub);
        let body = serde_json::json!({"model": "my-gpt4", "input": "hello"});
        let make_req = || {
            Request::builder()
                .method("POST")
                .uri("/v1/embeddings")
                .header("authorization", "Bearer sk-caller")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap()
        };

        // First request consumes the only RPM slot.
        let resp = run(build_router(state.clone()), make_req()).await;
        assert_eq!(resp.status(), StatusCode::OK);

        // Pre-fix: this would also return 200 (the gate didn't run on
        // /v1/embeddings). Post-fix: 429 because the rpm=1 cap is now
        // enforced uniformly via crate::quota::enforce.
        let resp = run(build_router(state.clone()), make_req()).await;
        assert_eq!(
            resp.status(),
            StatusCode::TOO_MANY_REQUESTS,
            "/v1/embeddings must enforce rate limits (issue #107); pre-fix it bypassed",
        );
        let body_bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        assert_eq!(v["error"]["type"], "rate_limit_exceeded");
    }

    // ---- regression coverage for api7/AISIX-Cloud#1116 --------------
    // Pre-fix the passthrough tunnel passed `None` to quota::enforce, so
    // a Model's inline rate_limit (and model-scope policies) never
    // applied to passthrough traffic — for provider endpoints with no
    // typed surface (e.g. video generation) the model limit was
    // unenforceable everywhere. Post-fix the top-level `model` field of
    // a JSON passthrough body is matched against the addressed
    // provider's configured Models and its limits reserved like the
    // typed endpoints.

    fn model_entry_with_rate_limit(
        name: &str,
        rate_limit: serde_json::Value,
    ) -> ResourceEntry<Model> {
        model_entry_named(name, "gpt-4o", rate_limit)
    }

    fn model_entry_named(
        display_name: &str,
        model_name: &str,
        rate_limit: serde_json::Value,
    ) -> ResourceEntry<Model> {
        let cfg = format!(
            r#"{{
                "display_name": "{display_name}",
                "provider": "openai",
                "model_name": "{model_name}",
                "provider_key_id": "{PK_ID}",
                "rate_limit": {rate_limit}
            }}"#
        );
        let model: Model = serde_json::from_str(&cfg).unwrap();
        ResourceEntry::new("model-id-1", model, 1)
    }

    #[tokio::test]
    async fn passthrough_enforces_model_rate_limit_from_body_model_field() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path(
                "/api/v1/services/aigc/video-generation/video-synthesis",
            ))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "output": {"task_id": "t-1", "task_status": "PENDING"}
            })))
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = new_snap(&upstream.uri());
        snap.models.insert(model_entry_with_rate_limit(
            "my-video-model",
            serde_json::json!({"rpm": 1}),
        ));
        snap.apikeys
            .insert(apikey_entry("sk-caller", &["my-video-model"]));
        let state = build_state(snap, hub);

        let body = serde_json::json!({
            "model": "my-video-model",
            "input": {"prompt": "a cardboard city at night"},
            "parameters": {"resolution": "720P"}
        });
        let make_req = || {
            Request::builder()
                .method("POST")
                .uri("/passthrough/openai/api/v1/services/aigc/video-generation/video-synthesis")
                .header("authorization", "Bearer sk-caller")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap()
        };

        // First request consumes the model's only RPM slot.
        let resp = run(build_router(state.clone()), make_req()).await;
        assert_eq!(resp.status(), StatusCode::OK);

        // Pre-fix: 200 again (model layer skipped on passthrough).
        // Post-fix: 429 — the body's `model` resolved the configured
        // Model and its rpm=1 cap now gates the tunnel.
        let resp = run(build_router(state.clone()), make_req()).await;
        assert_eq!(
            resp.status(),
            StatusCode::TOO_MANY_REQUESTS,
            "passthrough must enforce the body model's rate limit (api7/AISIX-Cloud#1116)",
        );
        let body_bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        assert_eq!(v["error"]["type"], "rate_limit_exceeded");
    }

    #[tokio::test]
    async fn passthrough_unregistered_or_absent_body_model_keeps_key_layer_only() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/anything"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"ok": true})))
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        // Model has rpm=1, but requests below never name it — its bucket
        // must stay untouched. The API key carries rpm=2 to pin that the
        // key layer still gates the tunnel (and 429s on the 3rd call).
        let snap = new_snap(&upstream.uri());
        snap.models.insert(model_entry_with_rate_limit(
            "my-video-model",
            serde_json::json!({"rpm": 1}),
        ));
        snap.apikeys.insert(apikey_entry_with_limits(
            "sk-caller",
            &["my-video-model"],
            Some(serde_json::json!({"rpm": 2})),
        ));
        let state = build_state(snap, hub);

        let make_req = |body: &'static str| {
            Request::builder()
                .method("POST")
                .uri("/passthrough/openai/anything")
                .header("authorization", "Bearer sk-caller")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap()
        };

        // Unregistered model name → no model-layer reservation, passes.
        let resp = run(
            build_router(state.clone()),
            make_req(r#"{"model":"not-a-configured-model","input":"x"}"#),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);

        // Non-JSON body → tolerated, no model-layer reservation.
        let resp = run(build_router(state.clone()), make_req("plain text body")).await;
        assert_eq!(resp.status(), StatusCode::OK);

        // Third call trips the KEY-level rpm=2 — the request-level
        // layers keep gating the tunnel exactly as before the fix.
        let resp = run(
            build_router(state.clone()),
            make_req(r#"{"model":"not-a-configured-model","input":"x"}"#),
        )
        .await;
        assert_eq!(
            resp.status(),
            StatusCode::TOO_MANY_REQUESTS,
            "key-level rate limit must keep applying to passthrough",
        );
    }

    /// A `model`-scope RateLimitPolicy row (no inline rate_limit on the
    /// Model) must gate the tunnel too — the policy path matches by the
    /// resolved entry id, which is only exercised when the body model
    /// lookup propagates it.
    #[tokio::test]
    async fn passthrough_enforces_model_scope_policy_from_body_model_field() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/anything"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"ok": true})))
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = new_snap(&upstream.uri());
        snap.models.insert(model_entry("my-video-model"));
        snap.rate_limit_policies.insert(ResourceEntry::new(
            "pol-1",
            serde_json::from_value(serde_json::json!({
                "name": "video-cap",
                "scope": "model",
                "scope_ref": "model-id-1",
                "window": "minute",
                "max_requests": 1
            }))
            .unwrap(),
            1,
        ));
        snap.apikeys
            .insert(apikey_entry("sk-caller", &["my-video-model"]));
        let state = build_state(snap, hub);

        let make_req = || {
            Request::builder()
                .method("POST")
                .uri("/passthrough/openai/anything")
                .header("authorization", "Bearer sk-caller")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"model":"my-video-model","input":"x"}"#))
                .unwrap()
        };

        let resp = run(build_router(state.clone()), make_req()).await;
        assert_eq!(resp.status(), StatusCode::OK);

        let resp = run(build_router(state.clone()), make_req()).await;
        assert_eq!(
            resp.status(),
            StatusCode::TOO_MANY_REQUESTS,
            "model-scope policy must gate passthrough via the body model",
        );
    }

    /// A schedule window that always matches "now" (every weekday, all
    /// day) so suspension is deterministic under the system clock.
    fn always_on_schedule() -> serde_json::Value {
        serde_json::json!([{
            "timezone": "UTC",
            "days_of_week": ["mon", "tue", "wed", "thu", "fri", "sat", "sun"],
            "start_time": "00:00",
            "end_time": "24:00"
        }])
    }

    /// A schedule window that can never match (a fixed past date).
    fn never_on_schedule() -> serde_json::Value {
        serde_json::json!([{
            "timezone": "UTC",
            "dates": ["2000-01-01"],
            "start_time": "00:00",
            "end_time": "24:00"
        }])
    }

    /// While inside a scheduled suspension window the policy reserves
    /// nothing; once the schedule no longer matches, enforcement resumes
    /// on the unchanged bucket (AISIX-Cloud#1104).
    #[tokio::test]
    async fn scheduled_suspension_pauses_policy_until_window_closes() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/anything"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"ok": true})))
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = new_snap(&upstream.uri());
        snap.models.insert(model_entry("my-video-model"));
        let policy_json = |schedules: serde_json::Value| {
            serde_json::json!({
                "name": "video-cap",
                "scope": "model",
                "scope_ref": "model-id-1",
                "window": "minute",
                "max_requests": 1,
                "schedules": schedules
            })
        };
        snap.rate_limit_policies.insert(ResourceEntry::new(
            "pol-1",
            serde_json::from_value(policy_json(always_on_schedule())).unwrap(),
            1,
        ));
        snap.apikeys
            .insert(apikey_entry("sk-caller", &["my-video-model"]));
        let state = build_state(snap, hub);

        let make_req = || {
            Request::builder()
                .method("POST")
                .uri("/passthrough/openai/anything")
                .header("authorization", "Bearer sk-caller")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"model":"my-video-model","input":"x"}"#))
                .unwrap()
        };

        // Suspended: max_requests=1 would reject the second call — both pass.
        for _ in 0..2 {
            let resp = run(build_router(state.clone()), make_req()).await;
            assert_eq!(
                resp.status(),
                StatusCode::OK,
                "suspended policy must not gate requests",
            );
        }

        // Swap in a schedule that no longer matches (as the loader does
        // when the window closes relative to a fresh evaluation).
        state
            .snapshot
            .load()
            .rate_limit_policies
            .insert(ResourceEntry::new(
                "pol-1",
                serde_json::from_value(policy_json(never_on_schedule())).unwrap(),
                2,
            ));

        let resp = run(build_router(state.clone()), make_req()).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let resp = run(build_router(state.clone()), make_req()).await;
        assert_eq!(
            resp.status(),
            StatusCode::TOO_MANY_REQUESTS,
            "policy must enforce again outside its suspension windows",
        );
    }

    /// The routing/ensemble per-target path (`reserve_model_only`)
    /// iterates the policy table independently of `reserve_layers`, so
    /// it must honor scheduled suspensions too (AISIX-Cloud#1104).
    #[tokio::test]
    async fn reserve_model_only_honors_scheduled_suspension() {
        let hub = Arc::new(Hub::new());
        let snap = new_snap("http://127.0.0.1:1");
        let model = model_entry("mg-member");
        let target = model.value.clone();
        snap.models.insert(model);
        let policy_json = |schedules: serde_json::Value| {
            serde_json::json!({
                "name": "member-cap",
                "scope": "model",
                "scope_ref": "model-id-1",
                "window": "minute",
                "max_requests": 1,
                "schedules": schedules
            })
        };
        snap.rate_limit_policies.insert(ResourceEntry::new(
            "pol-1",
            serde_json::from_value(policy_json(always_on_schedule())).unwrap(),
            1,
        ));
        let state = build_state(snap, hub);
        let auth = AuthenticatedKey {
            entry: Arc::new(ResourceEntry::new(
                "key-entry-1",
                serde_json::from_value::<aisix_core::ApiKey>(serde_json::json!({
                    "key_hash": "h",
                    "allowed_models": [],
                }))
                .unwrap(),
                1,
            )),
        };

        // Suspended: max_requests=1 would deny the second reservation
        // (pre_commit counts stick even when the reservation drops
        // uncommitted) — both succeed because nothing is reserved.
        for _ in 0..2 {
            let r =
                quota::reserve_model_only(&state, &auth, "mg-member", "model-id-1", &target).await;
            assert!(r.is_ok(), "suspended policy must reserve nothing");
        }

        state
            .snapshot
            .load()
            .rate_limit_policies
            .insert(ResourceEntry::new(
                "pol-1",
                serde_json::from_value(policy_json(never_on_schedule())).unwrap(),
                2,
            ));

        assert!(
            quota::reserve_model_only(&state, &auth, "mg-member", "model-id-1", &target)
                .await
                .is_ok()
        );
        assert!(
            quota::reserve_model_only(&state, &auth, "mg-member", "model-id-1", &target)
                .await
                .is_err(),
            "policy outside its windows must throttle the second reservation",
        );
    }

    /// The tunnel forwards bodies verbatim, so callers typically name the
    /// provider-native id (`model_name`), not the gateway alias
    /// (`display_name`). The limit must bind either way, and the bucket is
    /// keyed by the alias so tunnel and typed traffic share one budget.
    #[tokio::test]
    async fn passthrough_matches_provider_native_model_name_for_rate_limit() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/anything"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"ok": true})))
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = new_snap(&upstream.uri());
        snap.models.insert(model_entry_named(
            "ali-video-alias",
            "happyhorse-1.1-t2v",
            serde_json::json!({"rpm": 1}),
        ));
        snap.apikeys
            .insert(apikey_entry("sk-caller", &["ali-video-alias"]));
        let state = build_state(snap, hub);

        // Caller names the upstream id, not the alias — the alias's cap
        // must still bind.
        let make_req = || {
            Request::builder()
                .method("POST")
                .uri("/passthrough/openai/anything")
                .header("authorization", "Bearer sk-caller")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"model":"happyhorse-1.1-t2v","input":"x"}"#))
                .unwrap()
        };

        let resp = run(build_router(state.clone()), make_req()).await;
        assert_eq!(resp.status(), StatusCode::OK);

        let resp = run(build_router(state.clone()), make_req()).await;
        assert_eq!(
            resp.status(),
            StatusCode::TOO_MANY_REQUESTS,
            "provider-native model_name must resolve the Model's rate limit",
        );
    }

    /// A same-named Model registered under a DIFFERENT provider must not
    /// be charged for this tunnel's traffic: the body name only matches
    /// within the addressed provider.
    #[tokio::test]
    async fn passthrough_same_named_model_of_other_provider_is_not_limited() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/anything"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"ok": true})))
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = new_snap(&upstream.uri());
        // Credential-lending model for the addressed provider — unlimited.
        snap.models.insert(model_entry("openai-lender"));
        // Same body-name model under another provider, rpm=1. Its bucket
        // must stay untouched by /passthrough/openai traffic.
        let cross: Model = serde_json::from_str(&format!(
            r#"{{
                "display_name": "cross-model",
                "provider": "anthropic",
                "model_name": "cross-model",
                "provider_key_id": "{PK_ID}",
                "rate_limit": {{"rpm": 1}}
            }}"#
        ))
        .unwrap();
        snap.models
            .insert(ResourceEntry::new("model-id-cross", cross, 1));
        snap.apikeys
            .insert(apikey_entry("sk-caller", &["openai-lender", "cross-model"]));
        let state = build_state(snap, hub);

        let make_req = || {
            Request::builder()
                .method("POST")
                .uri("/passthrough/openai/anything")
                .header("authorization", "Bearer sk-caller")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"model":"cross-model","input":"x"}"#))
                .unwrap()
        };

        // Both calls pass: the anthropic model's rpm=1 bucket is never
        // drawn from by the openai tunnel.
        for _ in 0..2 {
            let resp = run(build_router(state.clone()), make_req()).await;
            assert_eq!(
                resp.status(),
                StatusCode::OK,
                "cross-provider same-named model must not gate this tunnel",
            );
        }
    }

    /// Regression for issue #108: streaming chat used to commit
    /// `0` tokens up front and never look at the upstream's terminal
    /// usage frame. TPM caps were silently bypassed for all streaming
    /// traffic. The fix: build_sse_stream now passes the largest
    /// `total_tokens` seen on any chunk to a callback that calls
    /// `Limiter::add_tokens_post_stream`, after the SSE stream
    /// completes. This test exercises that path end-to-end:
    ///
    /// 1. Issue one streaming request whose terminal SSE chunk
    ///    carries `usage.total_tokens = 1500`. Pre-fix this would
    ///    leave TPM at 0; post-fix TPM should be 1500.
    /// 2. Issue a second streaming request with the same key.
    ///    With TPM cap at 1000, this must 429 (not 200) — the
    ///    pre-emptive `tpm.is_exceeded` check on pre_commit catches
    ///    the over-shoot left by the previous request.
    #[tokio::test]
    async fn streaming_chat_tpm_cap_enforced_after_post_stream_commit_issue_108() {
        let upstream = MockServer::start().await;
        // Final SSE chunk carries usage attached to the stop chunk —
        // some OpenAI-compatible servers do this instead of OpenAI's
        // separate usage-only terminal frame. The bridge parses
        // `usage` off any chunk that has one, so both shapes commit
        // tokens (the separate-frame shape is covered by the #790
        // tests below).
        let sse = "\
data: {\"id\":\"up-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"up-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"up-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":500,\"completion_tokens\":1000,\"total_tokens\":1500}}\n\n\
data: [DONE]\n\n";
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse),
            )
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = seed_snapshot_with_limits(
            "my-gpt4",
            &["my-gpt4"],
            &upstream.uri(),
            serde_json::json!({"tpm": 1000}),
        );
        let state = build_state(snap, hub);
        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true
        });
        let make_req = || {
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", "Bearer sk-caller")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap()
        };

        // First streaming request succeeds. Drive the body to
        // completion so build_sse_stream's on_complete fires.
        let resp = run(build_router(state.clone()), make_req()).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let mut body_stream = resp.into_body().into_data_stream();
        while let Some(chunk) = body_stream.next().await {
            let _ = chunk.unwrap();
        }

        // Second request must 429 — TPM is now over-shot at 1500/1000.
        // Pre-fix TPM stayed at 0 and this would have been a 200.
        let resp = run(build_router(state.clone()), make_req()).await;
        assert_eq!(
            resp.status(),
            StatusCode::TOO_MANY_REQUESTS,
            "streaming chat must commit upstream tokens to TPM (issue #108); \
             pre-fix this returned 200 and the cap was bypassed",
        );
        let body_bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        assert_eq!(v["error"]["type"], "rate_limit_exceeded");
    }

    /// #790 (AISIX-Cloud): every OpenAI-protocol streaming leg now asks
    /// the upstream for the terminal usage frame by injecting
    /// `stream_options: {"include_usage": true}` when the client didn't
    /// set stream_options itself. Token telemetry used to record 0 for
    /// every streaming request whose client didn't ask. Three contracts,
    /// one mock (OpenAI's real include_usage shape — stop chunk without
    /// usage, then a usage-only frame with empty `choices`):
    ///  1. the outbound upstream body carries the injected stream_options;
    ///  2. the usage frame feeds the UsageEvent (500/1000, not 0/0);
    ///  3. the usage-only frame is stripped from the client-visible
    ///     stream — the client never asked for usage.
    #[tokio::test]
    async fn streaming_chat_injects_include_usage_and_strips_usage_frame_issue_790() {
        use aisix_obs::UsageSink;

        let upstream = MockServer::start().await;
        let sse = "\
data: {\"id\":\"up-790\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"up-790\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"up-790\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
data: {\"id\":\"up-790\",\"model\":\"gpt-4o\",\"choices\":[],\"usage\":{\"prompt_tokens\":500,\"completion_tokens\":1000,\"total_tokens\":1500}}\n\n\
data: [DONE]\n\n";
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse),
            )
            .mount(&upstream)
            .await;

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        let state = build_state(snap, hub).with_usage_sink(UsageSink::new(tx));

        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = run(build_router(state), req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let mut body_stream = resp.into_body().into_data_stream();
        let mut client_bytes = Vec::new();
        while let Some(chunk) = body_stream.next().await {
            client_bytes.extend_from_slice(&chunk.unwrap());
        }
        let client_body = String::from_utf8(client_bytes).unwrap();

        // (3) Content reaches the client; the unrequested usage frame
        // does not.
        assert!(client_body.contains("hi"));
        assert!(
            !client_body.contains("\"usage\""),
            "usage-only frame must be stripped when the client didn't \
             request stream_options.include_usage; got:\n{client_body}"
        );

        // (2) Telemetry carries the upstream-billed counts.
        let event = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("usage event was never emitted")
            .expect("sender dropped without sending");
        assert_eq!(event.prompt_tokens, 500);
        assert_eq!(event.completion_tokens, 1000);

        // (1) The outbound request asked for the usage frame.
        let reqs = upstream.received_requests().await.unwrap();
        let sent: serde_json::Value = serde_json::from_slice(&reqs[0].body).unwrap();
        assert_eq!(
            sent["stream_options"],
            serde_json::json!({"include_usage": true}),
            "streaming leg must inject stream_options.include_usage (#790)"
        );
    }

    /// Companion to the #790 test above: a client that asked for usage
    /// itself still receives the usage frame, and its stream_options
    /// passes through verbatim (no duplicate injection).
    #[tokio::test]
    async fn streaming_chat_forwards_usage_frame_when_client_asked_issue_790() {
        let upstream = MockServer::start().await;
        let sse = "\
data: {\"id\":\"up-790b\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"up-790b\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
data: {\"id\":\"up-790b\",\"model\":\"gpt-4o\",\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":7,\"total_tokens\":12}}\n\n\
data: [DONE]\n\n";
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse),
            )
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        let state = build_state(snap, hub);

        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true,
            "stream_options": {"include_usage": true}
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = run(build_router(state), req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let mut body_stream = resp.into_body().into_data_stream();
        let mut client_bytes = Vec::new();
        while let Some(chunk) = body_stream.next().await {
            client_bytes.extend_from_slice(&chunk.unwrap());
        }
        let client_body = String::from_utf8(client_bytes).unwrap();
        assert!(
            client_body.contains("\"prompt_tokens\":5"),
            "client asked for usage — the frame must be forwarded; got:\n{client_body}"
        );

        let reqs = upstream.received_requests().await.unwrap();
        let raw = String::from_utf8(reqs[0].body.clone()).unwrap();
        assert_eq!(
            raw.matches("stream_options").count(),
            1,
            "client-supplied stream_options must pass through exactly once"
        );
    }

    #[tokio::test]
    async fn rate_limit_rpm_returns_429_with_retry_after_header() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-up",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            })))
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = seed_snapshot_with_limits(
            "my-gpt4",
            &["my-gpt4"],
            &upstream.uri(),
            serde_json::json!({"rpm": 1}),
        );
        let state = build_state(snap, hub);
        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let make_req = || {
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", "Bearer sk-caller")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap()
        };

        // First request succeeds.
        let resp = run(build_router(state.clone()), make_req()).await;
        assert_eq!(resp.status(), StatusCode::OK);

        // Second request within the same minute trips rpm=1 → 429.
        let resp = run(build_router(state.clone()), make_req()).await;
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
        let retry = resp
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok())
            .expect("missing or malformed retry-after header");
        assert!(retry >= 1);
        let body_bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        assert_eq!(v["error"]["type"], "rate_limit_exceeded");
    }

    #[tokio::test]
    async fn rate_limit_tpm_blocks_after_token_commit_exhausts_window() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-up",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop"
                }],
                // Deliberately overshoot the TPM cap so the next
                // pre_commit observes an exhausted window.
                "usage": {"prompt_tokens": 10_000, "completion_tokens": 10_000, "total_tokens": 20_000}
            })))
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = seed_snapshot_with_limits(
            "my-gpt4",
            &["my-gpt4"],
            &upstream.uri(),
            serde_json::json!({"tpm": 1_000}),
        );
        let state = build_state(snap, hub);
        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let make_req = || {
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", "Bearer sk-caller")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap()
        };

        // First request goes through (pre-commit TPM is unchecked for an
        // empty bucket); usage counted at post-deduct overshoots the cap.
        let resp = run(build_router(state.clone()), make_req()).await;
        assert_eq!(resp.status(), StatusCode::OK);

        // Second request sees TPM > 1000 and rejects.
        let resp = run(build_router(state), make_req()).await;
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
        let body_bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        assert_eq!(v["error"]["type"], "rate_limit_exceeded");
    }

    #[tokio::test]
    async fn request_lifecycle_increments_metrics_counters() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-up",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}
            })))
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());

        let state = build_state(snap, hub);
        let metrics = state.metrics.clone();
        let app = build_router(state);

        // Pre-flight: counter family is absent until something writes.
        assert!(!metrics.render().contains("aisix_requests_total"));

        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);

        let rendered = metrics.render();
        assert!(rendered.contains("aisix_requests_total"));
        assert!(rendered.contains("provider=\"openai\""));
        assert!(rendered.contains("outcome=\"success\""));
        assert!(rendered.contains("aisix_tokens_consumed_total"));
        // 7 tokens were committed.
        assert!(
            rendered.contains("7"),
            "expected tokens counter at 7:\n{rendered}"
        );
    }

    #[tokio::test]
    async fn ratelimit_rejection_increments_ratelimit_counter() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-up",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            })))
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = seed_snapshot_with_limits(
            "my-gpt4",
            &["my-gpt4"],
            &upstream.uri(),
            serde_json::json!({"rpm": 1}),
        );
        let state = build_state(snap, hub);
        let metrics = state.metrics.clone();
        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let make_req = || {
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", "Bearer sk-caller")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap()
        };

        let _ = run(build_router(state.clone()), make_req()).await;
        let resp = run(build_router(state), make_req()).await;
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);

        let rendered = metrics.render();
        assert!(rendered.contains("aisix_ratelimit_rejections_total"));
        assert!(rendered.contains("scope=\"requests\""));
    }

    #[tokio::test]
    async fn cache_hit_short_circuits_upstream_and_sets_header() {
        // Wiremock that *only* satisfies one upstream call. If the cache
        // ever lets a second request through, the test fails with a 500.
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-up",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "cached"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            })))
            .expect(1) // hard expectation: exactly one upstream hit
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        // Cache gate opens only when an enabled policy exists in
        // snapshot. Without this seed step the test would 200 but
        // the cache header would be absent (policy-disabled path).
        seed_cache_policy(&snap, "test-cache");
        // Cache enabled — uses the default constructor.
        let state = build_state_with_cache(snap, hub);
        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let make_req = || {
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", "Bearer sk-caller")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap()
        };

        // First request — miss.
        let resp = run(build_router(state.clone()), make_req()).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers()
                .get("x-aisix-cache")
                .and_then(|v| v.to_str().ok()),
            Some("miss"),
        );

        // Second identical request — hit.
        let resp = run(build_router(state), make_req()).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers()
                .get("x-aisix-cache")
                .and_then(|v| v.to_str().ok()),
            Some("hit"),
        );
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["choices"][0]["message"]["content"], "cached");
    }

    /// Regression for #88. On a cache hit the DP must surface the
    /// cached response's prompt + completion tokens on a dedicated
    /// `cache_hit_saved_*` pair so cp-api can multiply by its pricing
    /// catalog server-side and report `cost_saved_usd` on `/usage`.
    /// Miss rows must keep the saved counters at zero.
    #[tokio::test]
    async fn cache_hit_emits_saved_token_counters_on_telemetry_event() {
        use aisix_obs::UsageSink;

        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-up",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "cached"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 7, "completion_tokens": 11, "total_tokens": 18}
            })))
            .expect(1)
            .mount(&upstream)
            .await;

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        seed_cache_policy(&snap, "test-cache");
        let state = build_state_with_cache(snap, hub).with_usage_sink(UsageSink::new(tx));

        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let make_req = || {
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", "Bearer sk-caller")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap()
        };

        // Miss: saved counters must be zero (the request paid the upstream).
        let _ = run(build_router(state.clone()), make_req()).await;
        let miss_event = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("miss event was never emitted")
            .expect("sender dropped");
        assert_eq!(miss_event.cache_status, "miss");
        assert_eq!(miss_event.cache_hit_saved_input_tokens, 0);
        assert_eq!(miss_event.cache_hit_saved_output_tokens, 0);

        // Hit: saved counters must mirror the cached response's usage.
        let _ = run(build_router(state), make_req()).await;
        let hit_event = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("hit event was never emitted")
            .expect("sender dropped");
        assert_eq!(hit_event.cache_status, "hit");
        assert_eq!(hit_event.cache_hit_saved_input_tokens, 7);
        assert_eq!(hit_event.cache_hit_saved_output_tokens, 11);
        // `prompt_tokens` keeps mirroring the cached usage too — the
        // existing dashboard rollups stay correct. The new field is
        // additive, not a substitute.
        assert_eq!(hit_event.prompt_tokens, 7);
        assert_eq!(hit_event.completion_tokens, 11);
    }

    #[tokio::test]
    async fn cache_miss_when_request_payload_differs() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-up",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            })))
            // Two distinct payloads → expect exactly two upstream calls.
            .expect(2)
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        seed_cache_policy(&snap, "test-cache");
        let state = build_state_with_cache(snap, hub);

        let body_a = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "first"}]
        });
        let body_b = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "second"}]
        });
        let mk = |body: &serde_json::Value| {
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", "Bearer sk-caller")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap()
        };

        let r1 = run(build_router(state.clone()), mk(&body_a)).await;
        let r2 = run(build_router(state), mk(&body_b)).await;
        for r in [r1, r2] {
            assert_eq!(r.status(), StatusCode::OK);
            assert_eq!(
                r.headers()
                    .get("x-aisix-cache")
                    .and_then(|v| v.to_str().ok()),
                Some("miss"),
            );
        }
    }

    #[tokio::test]
    async fn applies_to_model_does_not_cache_unmatched_model() {
        // Stage 3 contract: a `cache_policy` with
        // `applies_to = "model:<other>"` must NOT enable the cache for
        // requests targeting a different model. Three identical
        // requests should hit the upstream three times — none of them
        // gets cached.
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-up",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "always-fresh"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            })))
            .expect(3) // each call must reach the upstream
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        // The api key + model are named "my-gpt4"; the policy below
        // pins applies_to to a different model name so no request in
        // this test matches.
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        seed_cache_policy_with_applies_to(&snap, "scoped", "model:not-my-gpt4");
        let state = build_state_with_cache(snap, hub);

        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let make_req = || {
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", "Bearer sk-caller")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap()
        };

        // All three calls: 200, no `x-aisix-cache` header (policy gate
        // closed for this model). The wiremock `.expect(3)` above is
        // the load-bearing assertion — it fails the test at server
        // teardown if any call short-circuited via the cache.
        for _ in 0..3 {
            let resp = run(build_router(state.clone()), make_req()).await;
            assert_eq!(resp.status(), StatusCode::OK);
            assert!(
                resp.headers().get("x-aisix-cache").is_none(),
                "policy-gate-closed responses must not carry x-aisix-cache",
            );
        }
    }

    /// Issue #154 regression: a CachePolicy with `enabled: false`
    /// must NOT cache. The disabled policy must be filtered out by
    /// the find-first-enabled predicate at the chat.rs cache gate;
    /// every identical request must reach the upstream and the
    /// response must NOT carry an `x-aisix-cache` header (per the
    /// "policy-gate-closed = no header" contract pinned by the
    /// `applies_to_filters_out_unmatched_model` test above).
    #[tokio::test]
    async fn disabled_cache_policy_does_not_cache_and_emits_no_header() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-uncached",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "always-fresh"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            })))
            .expect(3) // each call must reach the upstream
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        // The disabled policy applies_to="all" (would match every
        // request), but `enabled: false` MUST cause the find-first-
        // enabled predicate to skip it.
        seed_cache_policy_disabled(&snap, "off-policy");
        let state = build_state_with_cache(snap, hub);

        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let make_req = || {
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", "Bearer sk-caller")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap()
        };

        // All three calls: 200, no `x-aisix-cache` header (policy
        // is disabled). wiremock's `.expect(3)` fails the test if
        // any call short-circuited via cache (= the disable flag
        // wasn't honored).
        for _ in 0..3 {
            let resp = run(build_router(state.clone()), make_req()).await;
            assert_eq!(resp.status(), StatusCode::OK);
            assert!(
                resp.headers().get("x-aisix-cache").is_none(),
                "disabled cache_policy must not emit x-aisix-cache header"
            );
        }
    }

    /// #519 B.8: a `backend: "redis"` policy on a DP without a redis
    /// cache must DISABLE caching for matching requests — both
    /// identical calls reach the upstream, neither carries an
    /// `x-aisix-cache` header, and telemetry reports
    /// `cache_status = "disabled"`. The pre-fix behavior (silent
    /// fallback to the node-local memory cache) would serve the
    /// second call from cache and fail wiremock's `.expect(2)`.
    #[tokio::test]
    async fn redis_backend_policy_without_redis_disables_caching() {
        use aisix_obs::UsageSink;

        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-up",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "fresh"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            })))
            .expect(2) // hard expectation: BOTH calls must pay the upstream
            .mount(&upstream)
            .await;

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        seed_cache_policy_with_backend(&snap, "redis-cache", "redis");
        // Default test state ships a memory cache but NO redis
        // instance — exactly the deployment the policy mismatches.
        let state = build_state_with_cache(snap, hub).with_usage_sink(UsageSink::new(tx));

        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let make_req = || {
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", "Bearer sk-caller")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap()
        };

        for _ in 0..2 {
            let resp = run(build_router(state.clone()), make_req()).await;
            assert_eq!(resp.status(), StatusCode::OK);
            assert!(
                resp.headers().get("x-aisix-cache").is_none(),
                "redis policy without a redis backend must not emit x-aisix-cache",
            );
            let event = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
                .await
                .expect("usage event was never emitted")
                .expect("sender dropped");
            assert_eq!(
                event.cache_status, "disabled",
                "unavailable backend must surface as cache_status=disabled",
            );
        }
    }

    /// #519 B.8 positive path: when the DP HAS a redis instance, a
    /// `backend: "redis"` policy must dispatch to it — not to the
    /// memory instance. A second MemoryCache stands in for redis
    /// (instance dispatch is under test, not the redis wire
    /// protocol): the second identical call is a cache hit, the
    /// entry lives in the redis instance, and the memory instance
    /// never saw the key.
    #[tokio::test]
    async fn redis_backend_policy_dispatches_to_redis_instance() {
        use aisix_cache::{Cache, CacheKey, MemoryCache};

        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-up",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "via-redis"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            })))
            .expect(1) // second call must be served from the redis instance
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        seed_cache_policy_with_backend(&snap, "redis-cache", "redis");

        let memory: Arc<dyn Cache> = Arc::new(MemoryCache::with_defaults());
        let redis_standin: Arc<dyn Cache> = Arc::new(MemoryCache::with_defaults());
        let mut state = build_state_with_cache(snap, hub);
        state.cache = Some(CacheBackends::new(
            memory.clone(),
            Some(redis_standin.clone()),
        ));

        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let make_req = || {
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", "Bearer sk-caller")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap()
        };

        let resp = run(build_router(state.clone()), make_req()).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers()
                .get("x-aisix-cache")
                .and_then(|v| v.to_str().ok()),
            Some("miss"),
        );

        let resp = run(build_router(state), make_req()).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers()
                .get("x-aisix-cache")
                .and_then(|v| v.to_str().ok()),
            Some("hit"),
        );

        // The entry must live in the redis instance and ONLY there —
        // a dispatch bug that wrote to the memory instance would
        // still produce a "hit" above, so pin the instance directly.
        // The stored key carries the gate's scope: policy id, purge
        // generation 0, and (scope defaults to `api_key`) the caller.
        let req: aisix_gateway::ChatFormat = serde_json::from_value(body).unwrap();
        let key = CacheKey::from_request(&req)
            .with_scope("cp-id-redis-cache", 0, Some("key-id-1"))
            .fingerprint();
        assert!(
            redis_standin.get(&key).await.unwrap().is_some(),
            "cache entry must be written to the policy's redis backend",
        );
        assert!(
            memory.get(&key).await.unwrap().is_none(),
            "memory instance must not be touched by a redis-backend policy",
        );
    }

    #[tokio::test]
    async fn applies_to_model_caches_matched_model() {
        // Counterpart to the negative test above: when the policy
        // `applies_to` matches the request's model, the cache gate
        // opens and the second identical request hits.
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-up",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "matched"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            })))
            .expect(1) // only one upstream hit; second call must come from cache
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        seed_cache_policy_with_applies_to(&snap, "scoped", "model:my-gpt4");
        let state = build_state_with_cache(snap, hub);
        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let make_req = || {
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", "Bearer sk-caller")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap()
        };

        let r1 = run(build_router(state.clone()), make_req()).await;
        assert_eq!(r1.status(), StatusCode::OK);
        assert_eq!(
            r1.headers()
                .get("x-aisix-cache")
                .and_then(|v| v.to_str().ok()),
            Some("miss"),
        );

        let r2 = run(build_router(state), make_req()).await;
        assert_eq!(r2.status(), StatusCode::OK);
        assert_eq!(
            r2.headers()
                .get("x-aisix-cache")
                .and_then(|v| v.to_str().ok()),
            Some("hit"),
        );
    }

    /// Build a `ResourceEntry<Model>` with a non-default id so the test
    /// can mount multiple Models in one snapshot. `pk_id` lets each
    /// model point at its own ProviderKey row — useful for routing
    /// tests that use multiple upstream MockServers.
    fn model_entry_with_id(id: &str, name: &str, pk_id: &str) -> ResourceEntry<Model> {
        let cfg = format!(
            r#"{{
                "display_name": "{name}",
                "provider": "openai",
                "model_name": "gpt-4o",
                "provider_key_id": "{pk_id}"
            }}"#
        );
        let model: Model = serde_json::from_str(&cfg).unwrap();
        ResourceEntry::new(id, model, 1)
    }

    fn pk_entry_with_id(pk_id: &str, api_base: &str) -> ResourceEntry<aisix_core::ProviderKey> {
        let cfg = format!(
            r#"{{"display_name":"openai-{pk_id}","secret":"sk-upstream","api_base":"{api_base}","provider":"openai","adapter":"openai"}}"#
        );
        let pk: aisix_core::ProviderKey = serde_json::from_str(&cfg).unwrap();
        ResourceEntry::new(pk_id, pk, 1)
    }

    /// Build a virtual routing Model that points at `targets` (other
    /// Model.display_name values) using the given strategy.
    fn routing_entry(
        name: &str,
        strategy: &str,
        targets: &[&str],
        retries: Option<u32>,
        max_fallbacks: Option<u32>,
        retry_on_429: Option<bool>,
    ) -> ResourceEntry<Model> {
        let target_objs: Vec<serde_json::Value> = targets
            .iter()
            .map(|t| serde_json::json!({"model": t}))
            .collect();
        let cfg = serde_json::json!({
            "display_name": name,
            "routing": {
                "strategy": strategy,
                "targets": target_objs,
                "retries": retries,
                "max_fallbacks": max_fallbacks,
                "retry_on_429": retry_on_429,
            }
        });
        let model: Model = serde_json::from_value(cfg).unwrap();
        ResourceEntry::new(format!("router-{name}"), model, 1)
    }

    #[tokio::test]
    async fn routing_failover_retries_to_second_target_when_first_5xxs() {
        let bad_upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(502).set_body_string("upstream down"))
            .mount(&bad_upstream)
            .await;

        let good_upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-good",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "fallback worked"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            })))
            .mount(&good_upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));

        let snap = AisixSnapshot::new();
        snap.provider_keys
            .insert(pk_entry_with_id("pk-bad", &bad_upstream.uri()));
        snap.provider_keys
            .insert(pk_entry_with_id("pk-good", &good_upstream.uri()));
        snap.models
            .insert(model_entry_with_id("m-bad", "primary", "pk-bad"));
        snap.models
            .insert(model_entry_with_id("m-good", "secondary", "pk-good"));
        snap.models.insert(routing_entry(
            "smart",
            "failover",
            &["primary", "secondary"],
            None,
            None,
            None,
        ));
        snap.apikeys.insert(apikey_entry("sk-caller", &["smart"]));

        let app = build_router(build_state(snap, hub));
        let body = serde_json::json!({
            "model": "smart",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let resp = run(app, req).await;
        let status = resp.status();
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        assert_eq!(
            status,
            StatusCode::OK,
            "{}",
            String::from_utf8_lossy(&bytes)
        );
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["choices"][0]["message"]["content"], "fallback worked");
    }

    #[tokio::test]
    async fn routing_propagates_4xx_without_attempting_fallback() {
        // First target returns 400 — caller mistake, no point trying
        // the second target. We assert the request fails 400 *and* the
        // second wiremock never sees a request.
        let bad_upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(400).set_body_string("invalid_request"))
            .expect(1)
            .mount(&bad_upstream)
            .await;

        let standby_upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            // Should never be hit; expect(0) enforces it on Drop.
            .respond_with(ResponseTemplate::new(200))
            .expect(0)
            .mount(&standby_upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));

        let snap = AisixSnapshot::new();
        snap.provider_keys
            .insert(pk_entry_with_id("pk-bad", &bad_upstream.uri()));
        snap.provider_keys
            .insert(pk_entry_with_id("pk-standby", &standby_upstream.uri()));
        snap.models
            .insert(model_entry_with_id("m-bad", "primary", "pk-bad"));
        snap.models
            .insert(model_entry_with_id("m-standby", "secondary", "pk-standby"));
        snap.models.insert(routing_entry(
            "smart",
            "failover",
            &["primary", "secondary"],
            None,
            None,
            None,
        ));
        snap.apikeys.insert(apikey_entry("sk-caller", &["smart"]));

        let app = build_router(build_state(snap, hub));
        let body = serde_json::json!({
            "model": "smart",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let resp = run(app, req).await;
        let status = resp.status();
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        assert_eq!(
            status,
            StatusCode::BAD_REQUEST,
            "{}",
            String::from_utf8_lossy(&bytes)
        );
    }

    /// A direct (non-group) model retries a transient upstream failure.
    ///
    /// The retry budget used to live only on `routing`, so a model without a
    /// model group was pinned at zero attempts-after-the-first: a single 502
    /// went straight back to the caller with no second try, no matter what
    /// the operator configured. Nothing could be configured — there was no
    /// field to set.
    #[tokio::test]
    async fn direct_model_retries_a_transient_failure_then_succeeds() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(502).set_body_string("upstream down"))
            .up_to_n_times(1)
            .with_priority(1)
            .expect(1)
            .mount(&upstream)
            .await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-ok",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "recovered"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            })))
            .with_priority(2)
            .expect(1)
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = new_snap(&upstream.uri());
        snap.models
            .insert(direct_model_entry("m-solo", "solo", "gpt-4o"));
        snap.apikeys.insert(apikey_entry("sk-caller", &["solo"]));
        let app = build_router(build_state(snap, hub));

        let body = serde_json::json!({
            "model": "solo",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let resp = run(app, req).await;
        let status = resp.status();
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        assert_eq!(
            status,
            StatusCode::OK,
            "the retry must recover the request: {}",
            String::from_utf8_lossy(&bytes)
        );
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["choices"][0]["message"]["content"], "recovered");
        // The `.expect(1)` on both mocks asserts exactly two upstream calls.
    }

    /// `Model.retries` overrides the group budget for that target only.
    /// `Some(0)` is an explicit opt-out and must not fall through to the
    /// group's larger budget.
    #[tokio::test]
    async fn model_retries_overrides_the_group_budget() {
        let bad_upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(502).set_body_string("upstream down"))
            // Exactly one attempt on this target despite `routing.retries: 5`.
            .expect(1)
            .mount(&bad_upstream)
            .await;

        let good_upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-good",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "fallback worked"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            })))
            .expect(1)
            .mount(&good_upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = AisixSnapshot::new();
        snap.provider_keys
            .insert(pk_entry_with_id("pk-bad", &bad_upstream.uri()));
        snap.provider_keys
            .insert(pk_entry_with_id("pk-good", &good_upstream.uri()));
        let mut no_retry: Model = serde_json::from_str(
            r#"{"display_name":"primary","provider":"openai","model_name":"gpt-4o",
                "provider_key_id":"pk-bad","retries":0}"#,
        )
        .unwrap();
        no_retry.retries = Some(0);
        snap.models.insert(ResourceEntry::new("m-bad", no_retry, 1));
        snap.models
            .insert(model_entry_with_id("m-good", "secondary", "pk-good"));
        snap.models.insert(routing_entry(
            "smart",
            "failover",
            &["primary", "secondary"],
            Some(5),
            None,
            None,
        ));
        snap.apikeys.insert(apikey_entry("sk-caller", &["smart"]));
        let app = build_router(build_state(snap, hub));

        let body = serde_json::json!({
            "model": "smart",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        // The `.expect(1)` on the bad upstream is the real assertion: the
        // target's own `retries: 0` beat the group's `retries: 5`.
    }

    #[tokio::test]
    async fn routing_retries_current_target_before_failover() {
        use aisix_obs::UsageSink;

        let flaky_upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(502).set_body_string("try again"))
            .expect(2)
            .mount(&flaky_upstream)
            .await;

        let good_upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-good",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "after retries"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            })))
            .expect(1)
            .mount(&good_upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));

        let snap = AisixSnapshot::new();
        snap.provider_keys
            .insert(pk_entry_with_id("pk-flaky", &flaky_upstream.uri()));
        snap.provider_keys
            .insert(pk_entry_with_id("pk-good", &good_upstream.uri()));
        snap.models
            .insert(model_entry_with_id("m-flaky", "primary", "pk-flaky"));
        snap.models
            .insert(model_entry_with_id("m-good", "secondary", "pk-good"));
        snap.models.insert(routing_entry(
            "smart",
            "failover",
            &["primary", "secondary"],
            Some(1),
            Some(1),
            None,
        ));
        snap.apikeys.insert(apikey_entry("sk-caller", &["smart"]));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let app = build_router(build_state(snap, hub).with_usage_sink(UsageSink::new(tx)));
        let body = serde_json::json!({
            "model": "smart",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let resp = run(app, req).await;
        let status = resp.status();
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        assert_eq!(
            status,
            StatusCode::OK,
            "{}",
            String::from_utf8_lossy(&bytes)
        );
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["choices"][0]["message"]["content"], "after retries");

        // Per #655 each upstream attempt emits its own UsageEvent, all
        // sharing `request_id`. Here: primary fails (initial), primary
        // fails again (retry), secondary succeeds (fallback) — 3 events.
        let mut events = Vec::new();
        for _ in 0..3 {
            let ev = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
                .await
                .expect("usage event was never emitted")
                .expect("sender dropped");
            events.push(ev);
        }
        events.sort_by_key(|e| e.attempt_index);
        let rid = events[0].request_id.clone();
        assert!(
            !rid.is_empty() && events.iter().all(|e| e.request_id == rid),
            "all attempts share the request_id (trace key)"
        );

        // AISIX-Cloud#790: every attempt carries the requested group
        // alias — model_id points at the per-attempt TARGET, so without
        // this the group name would appear nowhere.
        assert!(
            events.iter().all(|e| e.requested_model == "smart"),
            "every attempt records the requested group alias"
        );

        // initial attempt on `primary` failed with the upstream's 502
        assert_eq!(events[0].attempt_index, 0);
        assert_eq!(events[0].attempt_kind, "initial");
        assert_eq!(events[0].attempt_model, "primary");
        assert_eq!(
            events[0].model_id, "m-flaky",
            "failed attempt carries the TARGET's id"
        );
        assert_eq!(events[0].status_code, 502);
        assert_eq!(events[0].error_class, "upstream_status");
        assert_eq!(events[0].prompt_tokens, 0);
        assert_eq!(events[0].completion_tokens, 0);

        // retry on the SAME target, also failed
        assert_eq!(events[1].attempt_index, 1);
        assert_eq!(events[1].attempt_kind, "retry");
        assert_eq!(events[1].attempt_model, "primary");
        assert_eq!(events[1].model_id, "m-flaky");
        assert_eq!(events[1].status_code, 502);
        assert!(!events[1].error_class.is_empty());

        // fallback to `secondary` succeeded and carries the real tokens
        assert_eq!(events[2].attempt_index, 2);
        assert_eq!(events[2].attempt_kind, "fallback");
        assert_eq!(events[2].attempt_model, "secondary");
        // AISIX-Cloud#790: the winner records the TARGET's id, not the
        // group's — cp-api prices via model_id and group ids have no
        // pricing rows.
        assert_eq!(events[2].model_id, "m-good");
        assert_eq!(events[2].status_code, 200);
        assert_eq!(events[2].error_class, "");
        assert_eq!(events[2].prompt_tokens, 1);
        assert_eq!(events[2].completion_tokens, 1);
    }

    #[tokio::test]
    async fn streaming_routing_records_failed_initial_attempt() {
        use aisix_obs::UsageSink;

        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(502).set_body_string("stream unavailable"))
            .expect(1)
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));

        let snap = AisixSnapshot::new();
        snap.provider_keys
            .insert(pk_entry_with_id("pk-primary", &upstream.uri()));
        snap.models
            .insert(model_entry_with_id("m-primary", "primary", "pk-primary"));
        snap.models.insert(routing_entry(
            "smart",
            "failover",
            &["primary"],
            // Single-target group: there is nothing to fall over to, so the
            // default budget WOULD apply here. Pin it off — this test asserts
            // the attempt record for one failed try, not the retry policy.
            Some(0),
            None,
            None,
        ));
        snap.apikeys.insert(apikey_entry("sk-caller", &["smart"]));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let app = build_router(build_state(snap, hub).with_usage_sink(UsageSink::new(tx)));
        let body = serde_json::json!({
            "model": "smart",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);

        // Single target, `retries` unset (defaults to 0): exactly one
        // attempt, emitted as one per-attempt event (#655).
        let event = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("usage event was never emitted")
            .expect("sender dropped");
        assert_eq!(event.attempt_index, 0);
        assert_eq!(event.attempt_kind, "initial");
        assert_eq!(event.attempt_model, "primary");
        // AISIX-Cloud#790: failed streaming attempt carries the
        // TARGET's id + the requested group alias.
        assert_eq!(event.model_id, "m-primary");
        assert_eq!(event.requested_model, "smart");
        assert_eq!(event.status_code, 502);
        assert_eq!(event.error_class, "upstream_status");
        assert_eq!(event.prompt_tokens, 0);
        // No further events for this single-attempt request.
        if let Ok(Some(extra)) =
            tokio::time::timeout(std::time::Duration::from_millis(100), rx.recv()).await
        {
            panic!(
                "unexpected 2nd event: idx={} kind={} model={} status={} err={}",
                extra.attempt_index,
                extra.attempt_kind,
                extra.attempt_model,
                extra.status_code,
                extra.error_class
            );
        }
    }

    /// AISIX-Cloud#1119: the streaming path must honour `routing.retries`
    /// exactly like the non-streaming one. Before the fix the streaming
    /// loop walked targets once and never re-hit the same target, so a
    /// retryable failure fell straight over — the operator saw
    /// `initial → fallback` with the configured `retry #1` missing.
    /// `retries=1` on an always-502 primary must make TWO attempts on it
    /// before the secondary is tried.
    #[tokio::test]
    async fn streaming_routing_honors_same_target_retries() {
        use aisix_obs::UsageSink;

        let flaky_upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(502).set_body_string("try again"))
            // initial + one same-target retry. Pre-fix this was 1.
            .expect(2)
            .mount(&flaky_upstream)
            .await;

        let good_upstream = MockServer::start().await;
        let sse = "\
data: {\"id\":\"up-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"after retries\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"up-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
data: [DONE]\n\n";
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse),
            )
            .expect(1)
            .mount(&good_upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));

        let snap = AisixSnapshot::new();
        snap.provider_keys
            .insert(pk_entry_with_id("pk-flaky", &flaky_upstream.uri()));
        snap.provider_keys
            .insert(pk_entry_with_id("pk-good", &good_upstream.uri()));
        snap.models
            .insert(model_entry_with_id("m-flaky", "primary", "pk-flaky"));
        snap.models
            .insert(model_entry_with_id("m-good", "secondary", "pk-good"));
        snap.models.insert(routing_entry(
            "smart",
            "failover",
            &["primary", "secondary"],
            Some(1),
            Some(1),
            None,
        ));
        snap.apikeys.insert(apikey_entry("sk-caller", &["smart"]));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let app = build_router(build_state(snap, hub).with_usage_sink(UsageSink::new(tx)));
        let body = serde_json::json!({
            "model": "smart",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);

        // Drain the stream — the winning attempt's UsageEvent is emitted
        // by the end-of-stream Drop guard, so it isn't observable until
        // the body is fully consumed.
        let mut body_stream = resp.into_body().into_data_stream();
        let mut decoder = SseDecoder::new();
        let mut sse_events = Vec::new();
        while let Some(chunk) = body_stream.next().await {
            sse_events.extend(decoder.feed(chunk.unwrap().as_ref()));
        }
        assert!(sse_events.contains(&SseEvent::Done), "missing [DONE]");

        // initial (primary 502) → retry (primary 502) → fallback (secondary 200)
        let mut events = Vec::new();
        for _ in 0..3 {
            let ev = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
                .await
                .expect("usage event was never emitted")
                .expect("sender dropped");
            events.push(ev);
        }
        events.sort_by_key(|e| e.attempt_index);

        assert_eq!(events[0].attempt_kind, "initial");
        assert_eq!(events[0].attempt_model, "primary");
        assert_eq!(events[0].status_code, 502);

        // The attempt the operator reported missing in #1119.
        assert_eq!(
            events[1].attempt_kind, "retry",
            "same-target retry must precede fail-over"
        );
        assert_eq!(events[1].attempt_model, "primary");
        assert_eq!(events[1].model_id, "m-flaky");
        assert_eq!(events[1].status_code, 502);

        assert_eq!(events[2].attempt_kind, "fallback");
        assert_eq!(events[2].attempt_model, "secondary");
        assert_eq!(events[2].status_code, 200);
    }

    /// AISIX-Cloud#1119 / #1122: with a SINGLE target there is nothing to
    /// fail over to, so `routing.retries` is the only thing standing
    /// between a transient upstream blip and a failed request. The
    /// streaming path used to attempt once and give up.
    #[tokio::test]
    async fn streaming_routing_retries_single_target_before_failing() {
        use aisix_obs::UsageSink;

        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(502).set_body_string("transient"))
            // initial + two same-target retries. Pre-fix this was 1.
            .expect(3)
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));

        let snap = AisixSnapshot::new();
        snap.provider_keys
            .insert(pk_entry_with_id("pk-only", &upstream.uri()));
        snap.models
            .insert(model_entry_with_id("m-only", "primary", "pk-only"));
        snap.models.insert(routing_entry(
            "smart",
            "failover",
            &["primary"],
            Some(2),
            None,
            None,
        ));
        snap.apikeys.insert(apikey_entry("sk-caller", &["smart"]));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let app = build_router(build_state(snap, hub).with_usage_sink(UsageSink::new(tx)));
        let body = serde_json::json!({
            "model": "smart",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);

        let mut events = Vec::new();
        for _ in 0..3 {
            let ev = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
                .await
                .expect("usage event was never emitted")
                .expect("sender dropped");
            events.push(ev);
        }
        events.sort_by_key(|e| e.attempt_index);
        assert_eq!(events[0].attempt_kind, "initial");
        assert_eq!(events[1].attempt_kind, "retry");
        assert_eq!(events[2].attempt_kind, "retry");
        assert!(
            events.iter().all(|e| e.attempt_model == "primary"),
            "all attempts stay on the single configured target"
        );
    }

    #[tokio::test]
    async fn messages_routing_emits_per_attempt_events() {
        // Per #655 the /v1/messages family must emit one UsageEvent per
        // upstream attempt, just like /v1/chat/completions. A Model Group
        // whose primary 502s and secondary succeeds emits 2 events sharing
        // request_id, tagged inbound_protocol="anthropic".
        use aisix_obs::UsageSink;

        let bad_upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(502).set_body_string("upstream down"))
            .expect(1)
            .mount(&bad_upstream)
            .await;

        let good_upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-good",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "fallback worked"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            })))
            .expect(1)
            .mount(&good_upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));

        let snap = AisixSnapshot::new();
        snap.provider_keys
            .insert(pk_entry_with_id("pk-bad", &bad_upstream.uri()));
        snap.provider_keys
            .insert(pk_entry_with_id("pk-good", &good_upstream.uri()));
        snap.models
            .insert(model_entry_with_id("m-bad", "primary", "pk-bad"));
        snap.models
            .insert(model_entry_with_id("m-good", "secondary", "pk-good"));
        snap.models.insert(routing_entry(
            "smart",
            "failover",
            &["primary", "secondary"],
            // Unset on purpose: with a fallback target queued, the default
            // budget defers to it, so the sequence is initial -> fallback
            // with no same-target grinding in between.
            None,
            None,
            None,
        ));
        snap.apikeys.insert(apikey_entry("sk-caller", &["smart"]));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let app = build_router(build_state(snap, hub).with_usage_sink(UsageSink::new(tx)));
        let body = serde_json::json!({
            "model": "smart",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/messages")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let resp = run(app, req).await;
        let status = resp.status();
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        assert_eq!(
            status,
            StatusCode::OK,
            "{}",
            String::from_utf8_lossy(&bytes)
        );

        let mut events = Vec::new();
        for _ in 0..2 {
            let ev = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
                .await
                .expect("usage event was never emitted")
                .expect("sender dropped");
            events.push(ev);
        }
        events.sort_by_key(|e| e.attempt_index);
        let rid = events[0].request_id.clone();
        assert!(
            !rid.is_empty() && events.iter().all(|e| e.request_id == rid),
            "all attempts share the request_id (trace key)"
        );
        assert!(
            events.iter().all(|e| e.inbound_protocol == "anthropic"),
            "/v1/messages tags inbound_protocol=anthropic on every attempt"
        );
        // AISIX-Cloud#790: every attempt carries the requested group alias.
        assert!(
            events.iter().all(|e| e.requested_model == "smart"),
            "every attempt records the requested group alias"
        );

        // initial attempt on `primary` failed with the upstream's 502
        assert_eq!(events[0].attempt_index, 0);
        assert_eq!(events[0].attempt_kind, "initial");
        assert_eq!(events[0].attempt_model, "primary");
        assert_eq!(
            events[0].model_id, "m-bad",
            "failed attempt carries the TARGET's id"
        );
        assert_eq!(events[0].status_code, 502);
        assert_eq!(events[0].error_class, "upstream_status");
        assert_eq!(events[0].prompt_tokens, 0);
        assert_eq!(events[0].completion_tokens, 0);

        // fallback to `secondary` succeeded with real tokens
        assert_eq!(events[1].attempt_index, 1);
        assert_eq!(events[1].attempt_kind, "fallback");
        assert_eq!(events[1].attempt_model, "secondary");
        // AISIX-Cloud#790: the winner records the TARGET's id (pricing
        // resolves against it), not the group's.
        assert_eq!(events[1].model_id, "m-good");
        assert_eq!(events[1].status_code, 200);
        assert_eq!(events[1].error_class, "");
        assert_eq!(events[1].prompt_tokens, 1);
        assert_eq!(events[1].completion_tokens, 1);
    }

    #[tokio::test]
    async fn responses_routing_emits_per_attempt_events() {
        // Per #655 the /v1/responses family must emit one UsageEvent per
        // upstream attempt. A Model Group whose primary 502s and secondary
        // succeeds emits 2 events sharing request_id, inbound_protocol="openai".
        use aisix_obs::UsageSink;

        let bad_upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(502).set_body_string("upstream down"))
            .expect(1)
            .mount(&bad_upstream)
            .await;

        let good_upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "resp-good",
                "object": "response",
                "output": [],
                "usage": {"input_tokens": 1, "output_tokens": 1}
            })))
            .expect(1)
            .mount(&good_upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));

        let snap = AisixSnapshot::new();
        snap.provider_keys
            .insert(pk_entry_with_id("pk-bad", &bad_upstream.uri()));
        snap.provider_keys
            .insert(pk_entry_with_id("pk-good", &good_upstream.uri()));
        snap.models
            .insert(model_entry_with_id("m-bad", "primary", "pk-bad"));
        snap.models
            .insert(model_entry_with_id("m-good", "secondary", "pk-good"));
        snap.models.insert(routing_entry(
            "smart",
            "failover",
            &["primary", "secondary"],
            // Unset on purpose: with a fallback target queued, the default
            // budget defers to it, so the sequence is initial -> fallback
            // with no same-target grinding in between.
            None,
            None,
            None,
        ));
        snap.apikeys.insert(apikey_entry("sk-caller", &["smart"]));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let app = build_router(build_state(snap, hub).with_usage_sink(UsageSink::new(tx)));
        let body = serde_json::json!({
            "model": "smart",
            "input": "hi"
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let resp = run(app, req).await;
        let status = resp.status();
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        assert_eq!(
            status,
            StatusCode::OK,
            "{}",
            String::from_utf8_lossy(&bytes)
        );

        let mut events = Vec::new();
        for _ in 0..2 {
            let ev = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
                .await
                .expect("usage event was never emitted")
                .expect("sender dropped");
            events.push(ev);
        }
        events.sort_by_key(|e| e.attempt_index);
        let rid = events[0].request_id.clone();
        assert!(
            !rid.is_empty() && events.iter().all(|e| e.request_id == rid),
            "all attempts share the request_id (trace key)"
        );
        assert!(
            events.iter().all(|e| e.inbound_protocol == "openai"),
            "/v1/responses tags inbound_protocol=openai on every attempt"
        );
        // AISIX-Cloud#790: every attempt carries the requested group alias.
        assert!(
            events.iter().all(|e| e.requested_model == "smart"),
            "every attempt records the requested group alias"
        );

        // initial attempt on `primary` failed with the upstream's 502
        assert_eq!(events[0].attempt_index, 0);
        assert_eq!(events[0].attempt_kind, "initial");
        assert_eq!(events[0].attempt_model, "primary");
        assert_eq!(
            events[0].model_id, "m-bad",
            "failed attempt carries the TARGET's id"
        );
        assert_eq!(events[0].status_code, 502);
        assert_eq!(events[0].error_class, "upstream_status");
        assert_eq!(events[0].prompt_tokens, 0);

        // fallback to `secondary` succeeded with real tokens
        assert_eq!(events[1].attempt_index, 1);
        assert_eq!(events[1].attempt_kind, "fallback");
        assert_eq!(events[1].attempt_model, "secondary");
        // AISIX-Cloud#790: the winner records the TARGET's id (pricing
        // resolves against it), not the group's.
        assert_eq!(events[1].model_id, "m-good");
        assert_eq!(events[1].status_code, 200);
        assert_eq!(events[1].error_class, "");
        assert_eq!(events[1].prompt_tokens, 1);
        assert_eq!(events[1].completion_tokens, 1);
    }

    /// #641 parity: `/v1/messages` must honor `routing.retries` — re-hitting
    /// the SAME target before failing over, like chat.rs. A single-target group
    /// with `retries=1` and an always-502 target makes TWO attempts (initial +
    /// one same-target retry), classified `initial` then `retry`. Before the fix
    /// `/v1/messages` ignored `retries` and made only one attempt (so the
    /// upstream `.expect(2)` and the second event would never arrive).
    #[tokio::test]
    async fn messages_routing_honors_same_target_retries() {
        use aisix_obs::UsageSink;

        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(502).set_body_string("upstream down"))
            .expect(2)
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));

        let snap = AisixSnapshot::new();
        snap.provider_keys
            .insert(pk_entry_with_id("pk-bad", &upstream.uri()));
        snap.models
            .insert(model_entry_with_id("m-bad", "primary", "pk-bad"));
        snap.models.insert(routing_entry(
            "smart",
            "failover",
            &["primary"],
            Some(1),
            None,
            None,
        ));
        snap.apikeys.insert(apikey_entry("sk-caller", &["smart"]));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let app = build_router(build_state(snap, hub).with_usage_sink(UsageSink::new(tx)));
        let body = serde_json::json!({
            "model": "smart",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/messages")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);

        let mut events = Vec::new();
        for _ in 0..2 {
            let ev = tokio::time::timeout(std::time::Duration::from_millis(3000), rx.recv())
                .await
                .expect("two attempts (initial + retry) must each emit an event")
                .expect("sender dropped");
            events.push(ev);
        }
        events.sort_by_key(|e| e.attempt_index);
        assert_eq!(events[0].attempt_kind, "initial");
        assert_eq!(
            events[1].attempt_kind, "retry",
            "the same-target second attempt must be classified as a retry"
        );
        assert!(
            events.iter().all(|e| e.attempt_model == "primary"),
            "both attempts hit the SAME target (retry, not fallover)"
        );
        assert!(events.iter().all(|e| e.model_id == "m-bad"));
        assert!(events.iter().all(|e| e.status_code == 502));
        // upstream `.expect(2)` asserts exactly two upstream calls on Drop.
    }

    /// #641 parity for `/v1/responses` (Codex): same-target `routing.retries`
    /// before fail-over. Single-target group, `retries=1`, always-502 →
    /// initial + one retry, both hitting the same target.
    #[tokio::test]
    async fn responses_routing_honors_same_target_retries() {
        use aisix_obs::UsageSink;

        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(502).set_body_string("upstream down"))
            .expect(2)
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));

        let snap = AisixSnapshot::new();
        snap.provider_keys
            .insert(pk_entry_with_id("pk-bad", &upstream.uri()));
        snap.models
            .insert(model_entry_with_id("m-bad", "primary", "pk-bad"));
        snap.models.insert(routing_entry(
            "smart",
            "failover",
            &["primary"],
            Some(1),
            None,
            None,
        ));
        snap.apikeys.insert(apikey_entry("sk-caller", &["smart"]));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let app = build_router(build_state(snap, hub).with_usage_sink(UsageSink::new(tx)));
        let body = serde_json::json!({"model": "smart", "input": "hi"});
        let req = Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);

        let mut events = Vec::new();
        for _ in 0..2 {
            let ev = tokio::time::timeout(std::time::Duration::from_millis(3000), rx.recv())
                .await
                .expect("two attempts (initial + retry) must each emit an event")
                .expect("sender dropped");
            events.push(ev);
        }
        events.sort_by_key(|e| e.attempt_index);
        assert_eq!(events[0].attempt_kind, "initial");
        assert_eq!(
            events[1].attempt_kind, "retry",
            "the same-target second attempt must be classified as a retry"
        );
        assert!(
            events.iter().all(|e| e.attempt_model == "primary"),
            "both attempts hit the SAME target (retry, not fallover)"
        );
        assert!(events.iter().all(|e| e.status_code == 502));
    }

    /// AISIX-Cloud#790: a plain direct-model request (no routing group)
    /// records the client-sent name in `requested_model` and keeps the
    /// direct model's own id in `model_id` — on both the OpenAI and the
    /// Anthropic inbound protocols.
    #[tokio::test]
    async fn direct_model_requests_record_requested_model() {
        use aisix_obs::UsageSink;

        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-direct",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi there"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            })))
            .expect(2)
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let state = build_state(snap, hub).with_usage_sink(UsageSink::new(tx));

        // OpenAI protocol: /v1/chat/completions.
        let chat_req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::json!({
                    "model": "my-gpt4",
                    "messages": [{"role": "user", "content": "hi"}]
                })
                .to_string(),
            ))
            .unwrap();
        let resp = run(build_router(state.clone()), chat_req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let event = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("chat usage event was never emitted")
            .expect("sender dropped");
        assert_eq!(event.requested_model, "my-gpt4");
        // Direct request: target == requested entry, id unchanged.
        assert_eq!(event.model_id, "model-id-1");

        // Anthropic protocol: /v1/messages (cross-provider dispatch to
        // the same OpenAI upstream).
        let messages_req = Request::builder()
            .method("POST")
            .uri("/v1/messages")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::json!({
                    "model": "my-gpt4",
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "hi"}]
                })
                .to_string(),
            ))
            .unwrap();
        let resp = run(build_router(state), messages_req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let event = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("messages usage event was never emitted")
            .expect("sender dropped");
        assert_eq!(event.requested_model, "my-gpt4");
        assert_eq!(event.model_id, "model-id-1");
        assert_eq!(event.inbound_protocol, "anthropic");
    }

    #[tokio::test]
    async fn routing_can_retry_and_failover_on_429_when_enabled() {
        let ratelimited_upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(429).set_body_string("slow down"))
            .expect(2)
            .mount(&ratelimited_upstream)
            .await;

        let good_upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-good",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "429 fallback worked"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            })))
            .expect(1)
            .mount(&good_upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));

        let snap = AisixSnapshot::new();
        snap.provider_keys
            .insert(pk_entry_with_id("pk-429", &ratelimited_upstream.uri()));
        snap.provider_keys
            .insert(pk_entry_with_id("pk-good", &good_upstream.uri()));
        snap.models
            .insert(model_entry_with_id("m-429", "primary", "pk-429"));
        snap.models
            .insert(model_entry_with_id("m-good", "secondary", "pk-good"));
        snap.models.insert(routing_entry(
            "smart",
            "failover",
            &["primary", "secondary"],
            Some(1),
            Some(1),
            Some(true),
        ));
        snap.apikeys.insert(apikey_entry("sk-caller", &["smart"]));

        let app = build_router(build_state(snap, hub));
        let body = serde_json::json!({
            "model": "smart",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let resp = run(app, req).await;
        let status = resp.status();
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        assert_eq!(
            status,
            StatusCode::OK,
            "{}",
            String::from_utf8_lossy(&bytes)
        );
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["choices"][0]["message"]["content"], "429 fallback worked");
    }

    #[tokio::test]
    async fn routing_skips_target_in_runtime_cooldown() {
        let cooled_upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-cooled",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "should not be called"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            })))
            .expect(0)
            .mount(&cooled_upstream)
            .await;

        let good_upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-good",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "cooldown skipped"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            })))
            .expect(1)
            .mount(&good_upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));

        let snap = AisixSnapshot::new();
        snap.provider_keys
            .insert(pk_entry_with_id("pk-cooled", &cooled_upstream.uri()));
        snap.provider_keys
            .insert(pk_entry_with_id("pk-good", &good_upstream.uri()));
        snap.models
            .insert(model_entry_with_id("m-cooled", "primary", "pk-cooled"));
        snap.models
            .insert(model_entry_with_id("m-good", "secondary", "pk-good"));
        snap.models.insert(routing_entry(
            "smart",
            "failover",
            &["primary", "secondary"],
            None,
            None,
            None,
        ));
        snap.apikeys.insert(apikey_entry("sk-caller", &["smart"]));

        let state = build_state(snap, hub);
        state.runtime_status.mark_cooldown(
            "m-cooled",
            std::time::Duration::from_secs(30),
            "retryable_failure",
        );
        let app = build_router(state);
        let body = serde_json::json!({
            "model": "smart",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let resp = run(app, req).await;
        let status = resp.status();
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        assert_eq!(
            status,
            StatusCode::OK,
            "{}",
            String::from_utf8_lossy(&bytes)
        );
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["choices"][0]["message"]["content"], "cooldown skipped");
    }

    #[tokio::test]
    async fn routing_ignores_cooldown_when_it_would_empty_all_candidates() {
        let primary_upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-primary",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "cooldown fallback"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            })))
            .expect(1)
            .mount(&primary_upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));

        let snap = AisixSnapshot::new();
        snap.provider_keys
            .insert(pk_entry_with_id("pk-primary", &primary_upstream.uri()));
        snap.models
            .insert(model_entry_with_id("m-primary", "primary", "pk-primary"));
        snap.models.insert(routing_entry(
            "smart",
            "failover",
            &["primary"],
            None,
            None,
            None,
        ));
        snap.apikeys.insert(apikey_entry("sk-caller", &["smart"]));

        let state = build_state(snap, hub);
        state.runtime_status.mark_cooldown(
            "m-primary",
            std::time::Duration::from_secs(30),
            "retryable_failure",
        );
        let app = build_router(state);
        let body = serde_json::json!({
            "model": "smart",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let resp = run(app, req).await;
        let status = resp.status();
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        assert_eq!(
            status,
            StatusCode::OK,
            "{}",
            String::from_utf8_lossy(&bytes)
        );
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["choices"][0]["message"]["content"], "cooldown fallback");
    }

    #[tokio::test]
    async fn routing_retryable_failure_puts_target_into_cooldown_for_next_request() {
        let flaky_upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(502).set_body_string("temporary upstream failure"))
            .expect(1)
            .mount(&flaky_upstream)
            .await;

        let stable_upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-stable",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "stable fallback"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            })))
            .expect(2)
            .mount(&stable_upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));

        let snap = AisixSnapshot::new();
        snap.provider_keys
            .insert(pk_entry_with_id("pk-flaky", &flaky_upstream.uri()));
        snap.provider_keys
            .insert(pk_entry_with_id("pk-stable", &stable_upstream.uri()));
        snap.models
            .insert(model_entry_with_id("m-flaky", "primary", "pk-flaky"));
        snap.models
            .insert(model_entry_with_id("m-stable", "secondary", "pk-stable"));
        snap.models.insert(routing_entry(
            "smart",
            "failover",
            &["primary", "secondary"],
            Some(0),
            Some(1),
            None,
        ));
        snap.apikeys.insert(apikey_entry("sk-caller", &["smart"]));

        let state = build_state(snap, hub);
        let app = build_router(state.clone());
        let body = serde_json::json!({
            "model": "smart",
            "messages": [{"role": "user", "content": "first"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            state.runtime_status.status("m-flaky").status,
            RuntimeStatus::Cooldown
        );

        let app = build_router(state);
        let body = serde_json::json!({
            "model": "smart",
            "messages": [{"role": "user", "content": "second"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let resp = run(app, req).await;
        let status = resp.status();
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        assert_eq!(
            status,
            StatusCode::OK,
            "{}",
            String::from_utf8_lossy(&bytes)
        );
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["choices"][0]["message"]["content"], "stable fallback");
    }

    #[tokio::test]
    async fn routing_to_missing_target_returns_400() {
        // Routing references a Model that isn't in the snapshot — this
        // is a misconfiguration and should surface as a clean 400.
        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));

        let snap = AisixSnapshot::new();
        snap.models.insert(routing_entry(
            "smart",
            "failover",
            &["nonexistent"],
            None,
            None,
            None,
        ));
        snap.apikeys.insert(apikey_entry("sk-caller", &["smart"]));
        // No upstream provider_key needed — the routing target itself
        // is missing so dispatch fails before any provider lookup.

        let app = build_router(build_state(snap, hub));
        let body = serde_json::json!({
            "model": "smart",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn ratelimit_response_headers_are_injected_on_success() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-up",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            })))
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = seed_snapshot_with_limits(
            "my-gpt4",
            &["my-gpt4"],
            &upstream.uri(),
            serde_json::json!({"rpm": 100, "tpm": 50000}),
        );
        let app = build_router(build_state(snap, hub));

        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);

        let headers = resp.headers();
        assert!(
            headers.contains_key("x-ratelimit-limit-requests"),
            "missing x-ratelimit-limit-requests"
        );
        assert_eq!(
            headers
                .get("x-ratelimit-limit-requests")
                .and_then(|v| v.to_str().ok()),
            Some("100"),
        );
        assert!(
            headers.contains_key("x-ratelimit-limit-tokens"),
            "missing x-ratelimit-limit-tokens"
        );
        assert_eq!(
            headers
                .get("x-ratelimit-limit-tokens")
                .and_then(|v| v.to_str().ok()),
            Some("50000"),
        );
        // Remaining should be limit - 1 (one request consumed).
        assert_eq!(
            headers
                .get("x-ratelimit-remaining-requests")
                .and_then(|v| v.to_str().ok()),
            Some("99"),
        );
    }

    #[tokio::test]
    async fn input_guardrail_block_returns_422_and_skips_upstream() {
        // wiremock that fails the test if it's hit at all.
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200))
            .expect(0) // hard expectation — guardrail must short-circuit
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        let state = build_state(snap, hub);
        seed_guardrail(
            &state.snapshot,
            "g-input-block",
            r#"{"name":"input-guard","kind":"keyword","patterns":[{"kind":"literal","value":"forbidden-token"}]}"#,
        );
        let app = build_router(state);

        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "say the forbidden-token please"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "content_filter");
        // Per #153, the wire-level `error.message` MUST NOT carry
        // the matched-pattern detail. The previous assertion
        // `.contains("forbidden-token")` pinned the leaky behavior
        // (the literal value of the forbidden pattern showing up
        // in the caller-visible message). Redaction keeps the
        // matched literal in operator logs (`tracing`) only.
        // Per #519 B.4b the message DOES name the guardrail that
        // fired — operator-assigned metadata, not matched content.
        let message = v["error"]["message"].as_str().unwrap();
        assert!(
            !message.contains("forbidden-token"),
            "wire-level error.message must not leak the matched literal; got {message:?}"
        );
        assert_eq!(
            message,
            "request blocked by content policy (guardrail 'input-guard')"
        );
    }

    /// Regression: a guardrail-blocked request must record the resolved
    /// model_id on its telemetry event. Earlier the error path hard-coded
    /// model_id="" for every failure, which left the dashboard /logs
    /// "Guardrail blocks" tab showing an empty model column.
    #[tokio::test]
    async fn input_guardrail_block_records_resolved_model_id_in_telemetry() {
        use aisix_obs::UsageSink;

        // Capturing usage sink — we read the emitted event off the
        // receiver to assert telemetry shape, not just the HTTP response.
        let (tx, mut rx) = tokio::sync::mpsc::channel(8);

        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200))
            .expect(0)
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        let state = build_state(snap, hub);
        seed_guardrail(
            &state.snapshot,
            "g-input-block",
            r#"{"name":"input-guard","kind":"keyword","patterns":[{"kind":"literal","value":"forbidden-token"}]}"#,
        );
        let state = state.with_usage_sink(UsageSink::new(tx));
        let app = build_router(state);

        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "say the forbidden-token please"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);

        // The seeded model_entry uses the literal id "model-id-1"
        // (see lib.rs::model_entry). Pinning the exact value catches
        // regressions where the id silently becomes empty.
        let event = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("usage event was never emitted")
            .expect("sender dropped without sending");
        assert_eq!(event.model_id, "model-id-1");
        assert_eq!(event.status_code, 422);
        assert!(event.guardrail_blocked);
    }

    #[tokio::test]
    async fn output_guardrail_block_returns_422_after_upstream_runs() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-up",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "here is your secret-string"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            })))
            .expect(1) // upstream IS called; guardrail blocks the response
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        let state = build_state(snap, hub);
        seed_guardrail(
            &state.snapshot,
            "g-output-block",
            r#"{"name":"output-guard","kind":"keyword","hook_point":"output","patterns":[{"kind":"literal","value":"secret-string"}]}"#,
        );
        let app = build_router(state);

        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "anything"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "content_filter");
        // Per #153, the matched literal from the model's response
        // ("secret-string" — what the upstream returned and the
        // guardrail matched) MUST NOT appear in the caller-visible
        // error envelope. Echoing it would defeat the entire point
        // of an output guardrail: anyone who can trigger the rule
        // could extract the model's forbidden output via the error
        // message. This is the most security-critical assertion
        // for the whole guardrail surface.
        let message = v["error"]["message"].as_str().unwrap();
        assert!(
            !message.contains("secret-string"),
            "output guardrail leaked the matched literal back to the caller; got {message:?}"
        );
        // The full error envelope (any field) must also be clean —
        // future regressions might leak via a different field
        // (param/code) so check the whole serialized blob.
        let blob = serde_json::to_string(&v).unwrap();
        assert!(
            !blob.contains("secret-string"),
            "output guardrail leaked the matched literal in the envelope; got {blob}"
        );
        // #519 B.4b: the redacted message names the guardrail that fired.
        assert_eq!(
            message,
            "response blocked by content policy (guardrail 'output-guard')"
        );
    }

    /// Regression for #226: when an output-content-filter blocks a
    /// response that the upstream already produced, the telemetry event
    /// MUST carry the upstream-billed `prompt_tokens` /
    /// `completion_tokens` instead of zeroing them. Pre-fix the error
    /// path uniformly emitted zeros for every error variant — the
    /// "request never reached the upstream" assumption baked into the
    /// failure-path comment was wrong for the output-block case where
    /// the upstream HAS run and the provider has already charged.
    #[tokio::test]
    async fn output_guardrail_block_records_upstream_usage_in_telemetry() {
        use aisix_obs::UsageSink;

        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-blocked-1",
                "model": "gpt-4o-2024-08-06",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "leak the secret-string"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
            })))
            .expect(1)
            .mount(&upstream)
            .await;

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        let state = build_state(snap, hub);
        seed_guardrail(
            &state.snapshot,
            "g-output-block",
            r#"{"name":"output-guard","kind":"keyword","hook_point":"output","patterns":[{"kind":"literal","value":"secret-string"}]}"#,
        );
        let state = state.with_usage_sink(UsageSink::new(tx));
        let app = build_router(state);

        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "anything"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);

        let event = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("usage event was never emitted")
            .expect("sender dropped without sending");
        // The customer paid the provider for these tokens — telemetry
        // must reflect that, not silently drop them to 0.
        assert_eq!(
            event.prompt_tokens, 11,
            "output-block must preserve the upstream's prompt_tokens"
        );
        assert_eq!(
            event.completion_tokens, 7,
            "output-block must preserve the upstream's completion_tokens"
        );
        assert!(event.guardrail_blocked);
        assert_eq!(event.status_code, 422);
        assert_eq!(event.provider_request_id, "cmpl-blocked-1");
        assert_eq!(event.provider_model_version, "gpt-4o-2024-08-06");
        assert_eq!(event.finish_reason, "stop");
        // cache_status reflects the per-policy gate the request went
        // through; "disabled" here because the test seeds no
        // cache_policy. A regression that drops cache_status on the
        // output-block path would surface as empty-string here.
        assert_eq!(event.cache_status, "disabled");
    }

    /// Regression for ai-gateway#196 audit HIGH-1: streaming chat
    /// telemetry must fire even when the client disconnects mid-
    /// stream. Pre-fix, on_complete lived in a `yield`-following
    /// branch of the async_stream! body that only ran when the
    /// consumer pulled — a dropped consumer (axum aborting the
    /// response future) skipped it entirely, so the customer's
    /// upstream call billed but the gateway recorded zero events.
    /// Post-fix, on_complete fires from a Drop guard so cancellation
    /// still produces a usage_event (with whatever counts were
    /// captured up to disconnect, typically 0 if disconnect beat
    /// the upstream's `usage` chunk).
    #[tokio::test]
    async fn streaming_chat_telemetry_fires_on_client_disconnect() {
        use aisix_obs::UsageSink;

        let upstream = MockServer::start().await;
        // Use a slow drip so we can disconnect before [DONE] arrives.
        let sse = "\
data: {\"id\":\"cmpl-cancel-1\",\"model\":\"gpt-4o-2024-08-06\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"cmpl-cancel-1\",\"model\":\"gpt-4o-2024-08-06\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"cmpl-cancel-1\",\"model\":\"gpt-4o-2024-08-06\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
data: [DONE]\n\n";
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse),
            )
            .mount(&upstream)
            .await;

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        let state = build_state(snap, hub).with_usage_sink(UsageSink::new(tx));
        let app = build_router(state);

        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);

        // Read ONE chunk then drop the response body — simulates a
        // client that hung up before the upstream's terminal chunk.
        // The Drop guard inside build_sse_stream must still fire
        // on_complete for this disconnected request.
        let mut body_stream = resp.into_body().into_data_stream();
        let _first = body_stream.next().await;
        drop(body_stream);

        let event = tokio::time::timeout(std::time::Duration::from_secs(2), rx.recv())
            .await
            .expect("usage event was never emitted (Drop guard regression)")
            .expect("sender dropped without sending");
        // The exact token counts depend on how many chunks reached
        // the guard before disconnect — could be 0 (disconnect beat
        // the upstream emission entirely) or more. The contract is
        // simply that an event fires; counts are best-effort. Pin
        // the structural fields to confirm we didn't grab some
        // unrelated event.
        assert_eq!(
            event.status_code, CLIENT_CLOSED_REQUEST,
            "an abandoned stream must be recorded as a client cancel, not as a success"
        );
        assert!(!event.guardrail_blocked);
    }

    /// The counterpart to the test above: a stream the consumer reads to
    /// completion must stay `200`. Without this, the `reached_end` flag
    /// could be wired to something that is never set and every streamed
    /// request would silently be reported as abandoned.
    #[tokio::test]
    async fn streaming_chat_telemetry_reports_200_when_fully_consumed() {
        use aisix_obs::UsageSink;

        let upstream = MockServer::start().await;
        let sse = "\
data: {\"id\":\"cmpl-full\",\"model\":\"gpt-4o-2024-08-06\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"cmpl-full\",\"model\":\"gpt-4o-2024-08-06\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"cmpl-full\",\"model\":\"gpt-4o-2024-08-06\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
data: [DONE]\n\n";
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse),
            )
            .mount(&upstream)
            .await;

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        let state = build_state(snap, hub).with_usage_sink(UsageSink::new(tx));
        let app = build_router(state);

        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);

        // Drain to the end, the way a client that wants the whole answer does.
        let mut body_stream = resp.into_body().into_data_stream();
        while body_stream.next().await.is_some() {}

        let event = tokio::time::timeout(std::time::Duration::from_secs(2), rx.recv())
            .await
            .expect("usage event was never emitted")
            .expect("sender dropped without sending");
        assert_eq!(
            event.status_code, 200,
            "a fully consumed stream must not be reported as a client cancel"
        );
    }

    /// Same contract, but with an output guardrail attached — the shape that
    /// puts an awaiting end-of-stream scan between the upstream's last chunk
    /// and `[DONE]`. `reached_end` must be set before that scan, so pin the
    /// outcome with one configured.
    ///
    /// This fixes the placement contract; it cannot reproduce the race
    /// itself. A keyword guardrail scans locally, so its await completes
    /// immediately and a consumer cannot realistically be dropped inside it.
    /// The actual protection is that all five stream paths mark the flag at
    /// upstream EOF, ahead of any scan.
    #[tokio::test]
    async fn streaming_chat_with_output_guardrail_reports_200_when_fully_consumed() {
        use aisix_obs::UsageSink;

        let upstream = MockServer::start().await;
        let sse = "\
data: {\"id\":\"cmpl-guard\",\"model\":\"gpt-4o-2024-08-06\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"cmpl-guard\",\"model\":\"gpt-4o-2024-08-06\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"all clear\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"cmpl-guard\",\"model\":\"gpt-4o-2024-08-06\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
data: [DONE]\n\n";
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse),
            )
            .mount(&upstream)
            .await;

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        let state = build_state(snap, hub).with_usage_sink(UsageSink::new(tx));
        // The literal never appears in the response above, so the scan runs
        // to completion and allows — the stream is delivered in full.
        seed_guardrail(
            &state.snapshot,
            "g-eos-scan",
            r#"{"name":"eos-scan-guard","kind":"keyword","hook_point":"output","patterns":[{"kind":"literal","value":"never-present-literal"}]}"#,
        );
        let app = build_router(state);

        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);

        let mut body_stream = resp.into_body().into_data_stream();
        while body_stream.next().await.is_some() {}

        let event = tokio::time::timeout(std::time::Duration::from_secs(2), rx.recv())
            .await
            .expect("usage event was never emitted")
            .expect("sender dropped without sending");
        assert_eq!(
            event.status_code, 200,
            "a fully consumed stream with an output guardrail must not be reported as a cancel"
        );
    }

    const CANCEL_METRIC: &str = "aisix_proxy_client_cancelled_requests_total";

    /// A caller that hangs up before the response head exists must still
    /// leave a trace. Every endpoint logs and meters from the tail of its
    /// own handler, which a cancelled future never reaches — so such a
    /// request used to be absent from the access log, the usage events
    /// AND the metrics simultaneously. That made "the client says it sent
    /// N requests but the gateway only logged M" unaccountable, and it
    /// hid exactly the case operators care about: a caller giving up
    /// during a long time-to-first-token.
    #[tokio::test]
    async fn client_cancel_before_response_head_is_recorded() {
        let upstream = MockServer::start().await;
        // Outlives the patience window below, so the handler is still
        // awaiting the upstream when its future is dropped.
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_delay(std::time::Duration::from_secs(30))
                    .set_body_json(serde_json::json!({
                        "id": "cmpl-never",
                        "model": "gpt-4o",
                        "choices": []
                    })),
            )
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        let state = build_state(snap, hub);
        let metrics = state.metrics.clone();
        let app = build_router(state);

        assert!(!metrics.render().contains(CANCEL_METRIC));

        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        // Dropping the in-flight future is precisely what axum does when
        // the client's connection goes away before the handler produced a
        // response head.
        let outcome =
            tokio::time::timeout(std::time::Duration::from_millis(300), app.oneshot(req)).await;
        assert!(
            outcome.is_err(),
            "upstream answered too fast to model a cancel"
        );

        let rendered = metrics.render();
        assert!(
            rendered.contains(CANCEL_METRIC),
            "cancelled request left no metric: {rendered}"
        );
        assert!(
            rendered.contains("endpoint=\"/v1/chat/completions\""),
            "cancel metric lost its endpoint label: {rendered}"
        );
    }

    /// The guard must stay silent on the happy path. A completed request
    /// is already logged and metered by its own handler; counting it as a
    /// client cancel too would make the new series useless for alerting.
    #[tokio::test]
    async fn completed_request_is_not_counted_as_client_cancel() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-ok",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            })))
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        let state = build_state(snap, hub);
        let metrics = state.metrics.clone();
        let app = build_router(state);

        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);

        let rendered = metrics.render();
        assert!(
            !rendered.contains(CANCEL_METRIC),
            "a completed request was miscounted as a client cancel: {rendered}"
        );
    }

    /// A panicking handler drops the guard mid-unwind with `armed` still
    /// set, which looks identical to a cancel from `Drop`'s point of view.
    /// Recording it would invent a client disconnect that never happened and
    /// bury the panic under a benign 499, so the guard must stay silent and
    /// let the panic's own signal stand.
    #[test]
    fn cancel_guard_stays_silent_during_unwind() {
        let metrics = std::sync::Arc::new(aisix_obs::Metrics::new(false));
        let probe = metrics.clone();

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = ClientCancelGuard {
                armed: true,
                metrics: metrics.clone(),
                endpoint: "/v1/chat/completions",
                method: axum::http::Method::POST,
                uri: "/v1/chat/completions".parse().unwrap(),
                request_id: "req-unwind".to_string(),
                started: std::time::Instant::now(),
            };
            panic!("handler blew up");
        }));

        assert!(result.is_err(), "the test's own panic must have unwound");
        assert!(
            !probe.render().contains(CANCEL_METRIC),
            "a panicking handler was miscounted as a client cancel: {}",
            probe.render()
        );
    }

    /// A mid-stream hang-up is NOT a head-phase cancel. By the time SSE
    /// bytes flow the response head is committed, the handler has already
    /// written its access log, and the per-stream Drop guard emits the
    /// usage event (see `streaming_chat_telemetry_fires_on_client_disconnect`).
    /// Counting it here as well would report one disconnect twice under
    /// two different outcomes.
    #[tokio::test]
    async fn mid_stream_disconnect_is_not_counted_as_head_phase_cancel() {
        let upstream = MockServer::start().await;
        let sse = "\
data: {\"id\":\"cmpl-mid\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"cmpl-mid\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n\
data: [DONE]\n\n";
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse),
            )
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        let state = build_state(snap, hub);
        let metrics = state.metrics.clone();
        let app = build_router(state);

        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);

        // Read one chunk, then hang up mid-stream.
        let mut body_stream = resp.into_body().into_data_stream();
        let _first = body_stream.next().await;
        drop(body_stream);

        let rendered = metrics.render();
        assert!(
            !rendered.contains(CANCEL_METRIC),
            "mid-stream disconnect was double-counted as a head-phase cancel: {rendered}"
        );
    }

    /// Regression for #225: streaming chat must read the terminal SSE
    /// chunk's `usage` block and forward those counts into the
    /// telemetry event. Pre-fix the streaming path captured only
    /// `total_tokens` (for rate-limit accounting) and dropped
    /// `prompt_tokens` / `completion_tokens` — telemetry recorded zero
    /// for every streamed request even though the DP had the real
    /// counts in hand.
    #[tokio::test]
    async fn streaming_chat_telemetry_records_usage_from_terminal_chunk() {
        use aisix_obs::UsageSink;

        let upstream = MockServer::start().await;
        // OpenAI's stream_options.include_usage=true shape: the final
        // delta chunk before [DONE] carries a `usage` block.
        let sse = "\
data: {\"id\":\"cmpl-stream-1\",\"model\":\"gpt-4o-2024-08-06\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"cmpl-stream-1\",\"model\":\"gpt-4o-2024-08-06\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"cmpl-stream-1\",\"model\":\"gpt-4o-2024-08-06\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":13,\"completion_tokens\":4,\"total_tokens\":17}}\n\n\
data: [DONE]\n\n";
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse),
            )
            .mount(&upstream)
            .await;

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        let state = build_state(snap, hub).with_usage_sink(UsageSink::new(tx));
        let app = build_router(state);

        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true,
            "stream_options": {"include_usage": true}
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);

        // Drain the SSE body so build_sse_stream's on_complete fires.
        // Telemetry emission is wired to that callback; the channel
        // stays empty until the full stream has been consumed.
        let mut body_stream = resp.into_body().into_data_stream();
        while let Some(chunk) = body_stream.next().await {
            let _ = chunk.unwrap();
        }

        let event = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("usage event was never emitted")
            .expect("sender dropped without sending");
        assert_eq!(
            event.prompt_tokens, 13,
            "streaming telemetry must capture prompt_tokens from the terminal chunk's usage block"
        );
        assert_eq!(
            event.completion_tokens, 4,
            "streaming telemetry must capture completion_tokens from the terminal chunk"
        );
        assert_eq!(event.status_code, 200);
        assert!(!event.guardrail_blocked);
        assert_eq!(event.provider_request_id, "cmpl-stream-1");
        assert_eq!(event.provider_model_version, "gpt-4o-2024-08-06");
        assert_eq!(event.finish_reason, "stop");
    }

    #[tokio::test]
    async fn budget_exceeded_returns_429() {
        use crate::budget::BudgetClient;

        // cp-api stand-in: returns a deny decision for our key.
        // Wire shape mirrors cp-api's budgetCheckResponse — see
        // internal/cpapi/resources/budget_check.go (prd-09b rev 2 §5.5).
        let cp = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/dp/budget_check"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "allow": false,
                "fail_mode": "closed",
                "reason": {
                    "type": "billing_error",
                    "code": "budget_exceeded",
                    "message": "monthly cap exceeded",
                    "scope": "api_key",
                    "scope_ref": "ak-uuid",
                    "limit_usd": "10.00",
                    "spent_usd": "10.50",
                    "period": "month",
                    "period_resets_at": "2026-05-01T00:00:00Z",
                    "retry_after_seconds": 86400
                }
            })))
            .mount(&cp)
            .await;

        // Upstream chat endpoint must NOT be hit — the budget check
        // fires before dispatch.
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200))
            .expect(0)
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));

        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        let state = build_state(snap, hub).with_budget_client(Arc::new(BudgetClient::new(
            cp.uri(),
            reqwest::Client::new(),
        )));

        let app = build_router(state);
        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "billing_error");
        assert_eq!(v["error"]["code"], "budget_exceeded");
    }

    // ─── Cross-protocol × upstream matrix ─────────────────────────
    //
    // Closes the gap noted in earlier review: the per-bridge wiremock
    // tests prove each Bridge translates ChatFormat ↔ its wire shape,
    // and the proxy lib tests above prove `/v1/chat/completions` end-
    // to-end against an OpenAi upstream — but the *integration* of an
    // OpenAI-protocol inbound request hitting an Anthropic / Gemini /
    // DeepSeek upstream had zero coverage. These tests pin the full
    // path: client body parser → Hub.get(provider) → Bridge.chat[_stream]
    // → upstream → Bridge response decoder → renderer → wire bytes.

    const MATRIX_ANTHROPIC_PK_ID: &str = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa";
    const MATRIX_GOOGLE_PK_ID: &str = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb";
    const MATRIX_DEEPSEEK_PK_ID: &str = "cccccccc-cccc-cccc-cccc-cccccccccccc";
    const MATRIX_COHERE_PK_ID: &str = "dddddddd-dddd-dddd-dddd-dddddddddddd";

    fn anthropic_model_entry(name: &str) -> ResourceEntry<Model> {
        let cfg = format!(
            r#"{{
                "display_name": "{name}",
                "provider": "anthropic",
                "model_name": "claude-3-5-haiku-20241022",
                "provider_key_id": "{MATRIX_ANTHROPIC_PK_ID}"
            }}"#
        );
        ResourceEntry::new("model-anthropic-1", serde_json::from_str(&cfg).unwrap(), 1)
    }

    fn gemini_model_entry(name: &str) -> ResourceEntry<Model> {
        let cfg = format!(
            r#"{{
                "display_name": "{name}",
                "provider": "google",
                "model_name": "gemini-2.0-flash",
                "provider_key_id": "{MATRIX_GOOGLE_PK_ID}"
            }}"#
        );
        ResourceEntry::new("model-gemini-1", serde_json::from_str(&cfg).unwrap(), 1)
    }

    fn deepseek_model_entry(name: &str) -> ResourceEntry<Model> {
        let cfg = format!(
            r#"{{
                "display_name": "{name}",
                "provider": "deepseek",
                "model_name": "deepseek-chat",
                "provider_key_id": "{MATRIX_DEEPSEEK_PK_ID}"
            }}"#
        );
        ResourceEntry::new("model-deepseek-1", serde_json::from_str(&cfg).unwrap(), 1)
    }

    fn cohere_model_entry(name: &str) -> ResourceEntry<Model> {
        let cfg = format!(
            r#"{{
                "display_name": "{name}",
                "provider": "cohere",
                "model_name": "command-r",
                "provider_key_id": "{MATRIX_COHERE_PK_ID}"
            }}"#
        );
        ResourceEntry::new("model-cohere-1", serde_json::from_str(&cfg).unwrap(), 1)
    }

    fn matrix_pk_entry(
        id: &'static str,
        secret: &str,
        api_base: &str,
        provider: &str,
        adapter: &str,
    ) -> ResourceEntry<aisix_core::ProviderKey> {
        let cfg = format!(
            r#"{{"display_name":"matrix-up","secret":"{secret}","api_base":"{api_base}","provider":"{provider}","adapter":"{adapter}"}}"#
        );
        let pk: aisix_core::ProviderKey = serde_json::from_str(&cfg).unwrap();
        ResourceEntry::new(id, pk, 1)
    }

    fn matrix_anthropic_pk(api_base: &str) -> ResourceEntry<aisix_core::ProviderKey> {
        matrix_pk_entry(
            MATRIX_ANTHROPIC_PK_ID,
            "sk-ant-test",
            api_base,
            "anthropic",
            "anthropic",
        )
    }

    fn matrix_gemini_pk(api_base: &str) -> ResourceEntry<aisix_core::ProviderKey> {
        matrix_pk_entry(
            MATRIX_GOOGLE_PK_ID,
            "ya29-test",
            api_base,
            "google",
            "openai",
        )
    }

    fn matrix_deepseek_pk(api_base: &str) -> ResourceEntry<aisix_core::ProviderKey> {
        matrix_pk_entry(
            MATRIX_DEEPSEEK_PK_ID,
            "sk-deepseek",
            api_base,
            "deepseek",
            "openai",
        )
    }

    fn matrix_cohere_pk(api_base: &str) -> ResourceEntry<aisix_core::ProviderKey> {
        matrix_pk_entry(
            MATRIX_COHERE_PK_ID,
            "cohere-key",
            api_base,
            "cohere",
            "openai",
        )
    }

    /// (OpenAI inbound) × (Anthropic upstream) × (non-streaming).
    /// The most valuable cross-protocol cell — exercises real wire-shape
    /// translation in both directions inside `AnthropicBridge::chat`.
    #[tokio::test]
    async fn matrix_openai_in_anthropic_upstream_non_streaming() {
        use aisix_provider_anthropic::AnthropicBridge;

        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(header("x-api-key", "sk-ant-test"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "msg_01",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello from Claude!"}],
                "model": "claude-3-5-haiku-20241022",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 5, "output_tokens": 4}
            })))
            .mount(&upstream)
            .await;

        let snap = AisixSnapshot::new();
        snap.provider_keys
            .insert(matrix_anthropic_pk(&upstream.uri()));
        snap.models.insert(anthropic_model_entry("my-claude"));
        snap.apikeys
            .insert(apikey_entry("sk-caller", &["my-claude"]));
        let hub = Arc::new(Hub::new());
        hub.register_specialized("anthropic", Arc::new(AnthropicBridge::new()));
        let app = build_router(build_state(snap, hub));

        let body = serde_json::json!({
            "model": "my-claude",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let v: serde_json::Value =
            serde_json::from_slice(&to_bytes(resp.into_body(), 65536).await.unwrap()).unwrap();
        // OpenAI-shape wire on the way out.
        assert_eq!(v["object"], "chat.completion");
        assert_eq!(v["choices"][0]["message"]["role"], "assistant");
        assert_eq!(v["choices"][0]["message"]["content"], "Hello from Claude!");
        assert_eq!(v["choices"][0]["finish_reason"], "stop");
        assert_eq!(v["usage"]["prompt_tokens"], 5);
        assert_eq!(v["usage"]["completion_tokens"], 4);
    }

    /// (OpenAI inbound) × (Anthropic upstream) × (streaming).
    /// Pin the SSE event-stream translation: AnthropicBridge ingests
    /// typed Anthropic events (message_start / content_block_delta /
    /// message_delta / message_stop) and emits flat OpenAI deltas.
    #[tokio::test]
    async fn matrix_openai_in_anthropic_upstream_streaming() {
        use aisix_provider_anthropic::AnthropicBridge;

        let upstream = MockServer::start().await;
        let sse = "\
event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_01\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"claude-3-5-haiku-20241022\",\"stop_reason\":null,\"usage\":{\"input_tokens\":5,\"output_tokens\":0}}}\n\n\
event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n\
event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hel\"}}\n\n\
event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"lo\"}}\n\n\
event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n\
event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":2}}\n\n\
event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse),
            )
            .mount(&upstream)
            .await;

        let snap = AisixSnapshot::new();
        snap.provider_keys
            .insert(matrix_anthropic_pk(&upstream.uri()));
        snap.models.insert(anthropic_model_entry("my-claude"));
        snap.apikeys
            .insert(apikey_entry("sk-caller", &["my-claude"]));
        let hub = Arc::new(Hub::new());
        hub.register_specialized("anthropic", Arc::new(AnthropicBridge::new()));
        let app = build_router(build_state(snap, hub));

        let body = serde_json::json!({
            "model": "my-claude",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true,
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers()
                .get(axum::http::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok()),
            Some("text/event-stream"),
        );
        let body =
            String::from_utf8(to_bytes(resp.into_body(), 65536).await.unwrap().to_vec()).unwrap();
        // OpenAI-shape SSE deltas on the way out.
        assert!(
            body.contains("\"object\":\"chat.completion.chunk\""),
            "missing OpenAI chunk envelope in:\n{body}"
        );
        assert!(body.contains("\"content\":\"hel\""));
        assert!(body.contains("\"content\":\"lo\""));
        assert!(body.contains("\"finish_reason\":\"stop\""));
        assert!(body.contains("data: [DONE]"));
    }

    /// (OpenAI inbound) × (Gemini upstream). Gemini is served by the
    /// `Adapter::Openai` family bridge — cp-api stores the Gemini PK
    /// with `adapter: "openai"` and `api_base` pointing at Google's
    /// `/v1beta/openai` compat endpoint. The integration test pins
    /// that an inbound OpenAI request resolves through the family
    /// bridge and round-trips Gemini's OpenAI-shape response.
    #[tokio::test]
    async fn matrix_openai_in_gemini_upstream_non_streaming() {
        use aisix_core::Adapter;
        use aisix_provider_openai::OpenAiBridge;

        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-gemini",
                "model": "gemini-2.0-flash",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello from Gemini!"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 4, "completion_tokens": 5, "total_tokens": 9}
            })))
            .mount(&upstream)
            .await;

        let snap = AisixSnapshot::new();
        snap.provider_keys.insert(matrix_gemini_pk(&upstream.uri()));
        snap.models.insert(gemini_model_entry("my-gemini"));
        snap.apikeys
            .insert(apikey_entry("sk-caller", &["my-gemini"]));
        let hub = Arc::new(Hub::new());
        hub.register_family(Adapter::Openai, Arc::new(OpenAiBridge::new()));
        let app = build_router(build_state(snap, hub));

        let body = serde_json::json!({
            "model": "my-gemini",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let v: serde_json::Value =
            serde_json::from_slice(&to_bytes(resp.into_body(), 65536).await.unwrap()).unwrap();
        assert_eq!(v["choices"][0]["message"]["content"], "Hello from Gemini!");
        assert_eq!(v["usage"]["total_tokens"], 9);
    }

    /// (OpenAI inbound) × (DeepSeek upstream). DeepSeek is served by
    /// the `Adapter::Openai` family bridge — cp-api stores the
    /// DeepSeek PK with `adapter: "openai"` and `api_base` pointing
    /// at `https://api.deepseek.com`. The integration test pins
    /// that an inbound OpenAI request resolves through the family
    /// bridge and round-trips DeepSeek's OpenAI-shape response.
    #[tokio::test]
    async fn matrix_openai_in_deepseek_upstream_non_streaming() {
        use aisix_core::Adapter;
        use aisix_provider_openai::OpenAiBridge;

        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(header("authorization", "Bearer sk-deepseek"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-deepseek",
                "model": "deepseek-chat",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello from DeepSeek!"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 6, "completion_tokens": 7, "total_tokens": 13}
            })))
            .mount(&upstream)
            .await;

        let snap = AisixSnapshot::new();
        snap.provider_keys
            .insert(matrix_deepseek_pk(&upstream.uri()));
        snap.models.insert(deepseek_model_entry("my-deepseek"));
        snap.apikeys
            .insert(apikey_entry("sk-caller", &["my-deepseek"]));
        let hub = Arc::new(Hub::new());
        hub.register_family(Adapter::Openai, Arc::new(OpenAiBridge::new()));
        let app = build_router(build_state(snap, hub));

        let body = serde_json::json!({
            "model": "my-deepseek",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let v: serde_json::Value =
            serde_json::from_slice(&to_bytes(resp.into_body(), 65536).await.unwrap()).unwrap();
        assert_eq!(
            v["choices"][0]["message"]["content"],
            "Hello from DeepSeek!"
        );
        assert_eq!(v["usage"]["total_tokens"], 13);
    }

    /// (OpenAI inbound) × (Cohere chat-compat upstream). Cohere serves
    /// an OpenAI-shape envelope at `/compatibility/v1/chat/completions`
    /// per <https://docs.cohere.com/reference/chat>; cp-api stores the
    /// Cohere PK with `adapter: "openai"` and `api_base` pointing at
    /// `https://api.cohere.com/compatibility/v1`. The integration test
    /// pins that an inbound OpenAI request resolves through the
    /// `Adapter::Openai` family bridge (no specialized "cohere"
    /// registration in this Hub) and round-trips Cohere's OpenAI-shape
    /// response.
    ///
    /// Backfills coverage lost when the #379 clean cut deleted the
    /// `cohere_chat_compat_round_trips_openai_envelope` unit test
    /// (which exercised `OpenAiBridge::with_name("cohere")`, a code
    /// path that no longer exists).
    #[tokio::test]
    async fn matrix_openai_in_cohere_chat_compat_non_streaming() {
        use aisix_core::Adapter;
        use aisix_provider_openai::OpenAiBridge;

        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(header("authorization", "Bearer cohere-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-cohere",
                "model": "command-r",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello from Cohere!"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}
            })))
            .mount(&upstream)
            .await;

        let snap = AisixSnapshot::new();
        // PK `api_base` points at the wiremock root the way cp-api's
        // adapter_map points real Cohere PKs at `…/compatibility/v1`.
        snap.provider_keys.insert(matrix_cohere_pk(&upstream.uri()));
        snap.models.insert(cohere_model_entry("my-cohere"));
        snap.apikeys
            .insert(apikey_entry("sk-caller", &["my-cohere"]));
        let hub = Arc::new(Hub::new());
        // Family-only registration — NO `register_specialized("cohere", …)`.
        // The whole point of the test is to prove the family bridge
        // serves Cohere chat-compat without a vendor-specific entry.
        hub.register_family(Adapter::Openai, Arc::new(OpenAiBridge::new()));
        let app = build_router(build_state(snap, hub));

        let body = serde_json::json!({
            "model": "my-cohere",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let v: serde_json::Value =
            serde_json::from_slice(&to_bytes(resp.into_body(), 65536).await.unwrap()).unwrap();
        assert_eq!(v["choices"][0]["message"]["content"], "Hello from Cohere!");
        assert_eq!(v["usage"]["total_tokens"], 7);
    }

    // ---------------------------------------------------------------
    // Ensemble dispatch glue (feat/ensemble-model).
    //
    // The pure fan-out / synthesis logic is covered by the unit tests
    // in `ensemble.rs` with a mock caller. This e2e test exercises the
    // chat.rs wiring end-to-end through the real HTTP handler +
    // ProxyModelCaller + bridge: an ensemble model fans out to two
    // panel members and a judge over real (wiremock) upstreams, and the
    // client receives the judge's synthesized answer rendered under the
    // requested ensemble model name.
    // ---------------------------------------------------------------

    /// A direct OpenAI model with a caller-chosen id + upstream model
    /// name, sharing the single test ProviderKey. Distinct `model_name`
    /// values let the wiremock body-matchers tell panel calls from the
    /// judge call apart.
    fn direct_model_entry(id: &str, name: &str, upstream_model: &str) -> ResourceEntry<Model> {
        let cfg = format!(
            r#"{{
                "display_name": "{name}",
                "provider": "openai",
                "model_name": "{upstream_model}",
                "provider_key_id": "{PK_ID}"
            }}"#
        );
        let model: Model = serde_json::from_str(&cfg).unwrap();
        ResourceEntry::new(id, model, 1)
    }

    /// An ensemble model referencing `panel` display names + a `judge`
    /// display name. Carries no provider/model_name of its own.
    fn ensemble_model_entry(
        id: &str,
        name: &str,
        panel: &[&str],
        judge: &str,
    ) -> ResourceEntry<Model> {
        let panel_json = panel
            .iter()
            .map(|m| format!(r#"{{"model":"{m}"}}"#))
            .collect::<Vec<_>>()
            .join(",");
        let cfg = format!(
            r#"{{
                "display_name": "{name}",
                "ensemble": {{
                    "panel": [{panel_json}],
                    "judge": {{"model": "{judge}"}}
                }}
            }}"#
        );
        let model: Model = serde_json::from_str(&cfg).unwrap();
        ResourceEntry::new(id, model, 1)
    }

    fn direct_model_entry_rl(
        id: &str,
        name: &str,
        upstream_model: &str,
        rate_limit: serde_json::Value,
    ) -> ResourceEntry<Model> {
        // `retries: 0` — the rate-limit tests that use this helper assert on
        // reservation accounting, and want one upstream call per dispatch.
        // Under the default budget a single mocked failure would be retried
        // into a success and the assertion would test nothing.
        let cfg = format!(
            r#"{{
                "display_name": "{name}",
                "provider": "openai",
                "model_name": "{upstream_model}",
                "provider_key_id": "{PK_ID}",
                "retries": 0,
                "rate_limit": {rate_limit}
            }}"#
        );
        let model: Model = serde_json::from_str(&cfg).unwrap();
        ResourceEntry::new(id, model, 1)
    }

    /// #620: a panel member must reserve its OWN model rate limit during
    /// fan-out. A self-ensemble of two copies of a model capped at
    /// `concurrency: 1` can only run one copy at a time, so the second is
    /// dropped and the panel falls below `min_responses` (which defaults to 2
    /// for a two-member panel) → 502. Before the per-target reservation both
    /// copies ran and the request returned 200.
    #[tokio::test]
    async fn ensemble_panel_member_concurrency_limit_enforced_per_target() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-panel",
                "model": "panel-upstream",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "a panel candidate answer"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 5, "completion_tokens": 11, "total_tokens": 16}
            })))
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));

        let snap = new_snap(&upstream.uri());
        // Panel member capped at one concurrent call.
        snap.models.insert(direct_model_entry_rl(
            "m-capped",
            "capped",
            "panel-upstream",
            serde_json::json!({ "concurrency": 1 }),
        ));
        snap.models
            .insert(direct_model_entry("m-judge", "judge-m", "judge-upstream"));
        // Self-ensemble: the SAME capped model twice → two concurrent calls
        // contend for one concurrency:1 bucket. Default min_responses = 2.
        snap.models.insert(ensemble_model_entry(
            "m-council",
            "council",
            &["capped", "capped"],
            "judge-m",
        ));
        snap.apikeys.insert(apikey_entry("sk-caller", &["council"]));
        let app = build_router(build_state(snap, hub));

        let body = serde_json::json!({
            "model": "council",
            "messages": [{"role": "user", "content": "what is the best answer?"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = run(app, req).await;
        // Only one copy acquires the concurrency:1 slot; the other is dropped,
        // leaving one panel response < min_responses(2) → insufficient panel.
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
    }

    /// #620: the per-member reservation must RELEASE on success (commit), not
    /// leak the slot. Two sequential requests through a single-member panel
    /// whose model is capped at `concurrency: 1` must BOTH succeed — if the
    /// first request leaked the slot, the second would be starved (502).
    #[tokio::test]
    async fn ensemble_panel_member_concurrency_slot_released_after_success() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(wiremock::matchers::body_string_contains("Answer 1:"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-judge",
                "model": "judge-upstream",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "synthesized final answer"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 30, "completion_tokens": 7, "total_tokens": 37}
            })))
            .with_priority(1)
            .mount(&upstream)
            .await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-panel",
                "model": "panel-upstream",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "a panel candidate answer"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 5, "completion_tokens": 11, "total_tokens": 16}
            })))
            .with_priority(2)
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));

        let snap = new_snap(&upstream.uri());
        snap.models.insert(direct_model_entry_rl(
            "m-capped",
            "capped",
            "panel-upstream",
            serde_json::json!({ "concurrency": 1 }),
        ));
        snap.models
            .insert(direct_model_entry("m-judge", "judge-m", "judge-upstream"));
        // Single-member panel → min_responses defaults to 1.
        snap.models.insert(ensemble_model_entry(
            "m-council",
            "council",
            &["capped"],
            "judge-m",
        ));
        snap.apikeys.insert(apikey_entry("sk-caller", &["council"]));
        let app = build_router(build_state(snap, hub));

        let make_req = || {
            let body = serde_json::json!({
                "model": "council",
                "messages": [{"role": "user", "content": "what is the best answer?"}]
            });
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", "Bearer sk-caller")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap()
        };
        // First request acquires + releases the capped model's only slot.
        assert_eq!(run(app.clone(), make_req()).await.status(), StatusCode::OK);
        // Second request must find the slot free (released on commit).
        assert_eq!(run(app, make_req()).await.status(), StatusCode::OK);
    }

    /// #620: the STREAMING judge bypasses `ProxyModelCaller::call` (it streams
    /// via `build_sse_stream`), so its own rate limit must be reserved on the
    /// streaming path too. With the judge capped at `rpm: 1`, the first streamed
    /// ensemble request consumes the judge's only request slot (200); the second
    /// finds it exhausted and fails before opening the stream (429).
    #[tokio::test]
    async fn ensemble_streaming_judge_rate_limit_enforced() {
        let upstream = MockServer::start().await;
        let judge_sse = "\
data: {\"id\":\"cmpl-judge\",\"model\":\"judge-upstream\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"final\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"cmpl-judge\",\"model\":\"judge-upstream\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":30,\"completion_tokens\":7,\"total_tokens\":37}}\n\n\
data: [DONE]\n\n";
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(wiremock::matchers::body_string_contains("Answer 1:"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(judge_sse),
            )
            .with_priority(1)
            .mount(&upstream)
            .await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-panel",
                "model": "panel-upstream",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "a panel candidate answer"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 5, "completion_tokens": 11, "total_tokens": 16}
            })))
            .with_priority(2)
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = new_snap(&upstream.uri());
        snap.models
            .insert(direct_model_entry("m-panel-a", "panel-a", "panel-upstream"));
        snap.models
            .insert(direct_model_entry("m-panel-b", "panel-b", "panel-upstream"));
        // Judge capped at one request per minute.
        snap.models.insert(direct_model_entry_rl(
            "m-judge",
            "judge-m",
            "judge-upstream",
            serde_json::json!({ "rpm": 1 }),
        ));
        snap.models.insert(ensemble_model_entry(
            "m-council",
            "council",
            &["panel-a", "panel-b"],
            "judge-m",
        ));
        snap.apikeys.insert(apikey_entry("sk-caller", &["council"]));
        let app = build_router(build_state(snap, hub));

        let make_req = || {
            let body = serde_json::json!({
                "model": "council",
                "messages": [{"role": "user", "content": "what is the best answer?"}],
                "stream": true
            });
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", "Bearer sk-caller")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap()
        };
        // First streamed request consumes the judge's only rpm slot.
        assert_eq!(run(app.clone(), make_req()).await.status(), StatusCode::OK);
        // Second finds the judge rpm exhausted → 429 before the stream opens.
        assert_eq!(
            run(app, make_req()).await.status(),
            StatusCode::TOO_MANY_REQUESTS
        );
    }

    /// #620 (audit M-1): the per-member `commit_tokens` must accrue to the
    /// member's OWN `model:` TPM bucket. Panel member capped at tpm:10 returns
    /// 16 tokens: request 1 succeeds and commits 16 (overshoot allowed for the
    /// in-flight call); request 2's pre-commit sees tpm 16 ≥ 10 and refuses, so
    /// the single-member panel falls below min_responses → 502. If the member's
    /// tokens were never committed (or committed to the wrong bucket), tpm would
    /// stay 0 and request 2 would also succeed.
    #[tokio::test]
    async fn ensemble_panel_member_token_commit_accrues_to_model_bucket() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(wiremock::matchers::body_string_contains("Answer 1:"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-judge",
                "model": "judge-upstream",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "synthesized final answer"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 30, "completion_tokens": 7, "total_tokens": 37}
            })))
            .with_priority(1)
            .mount(&upstream)
            .await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-panel",
                "model": "panel-upstream",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "a panel candidate answer"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 5, "completion_tokens": 11, "total_tokens": 16}
            })))
            .with_priority(2)
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = new_snap(&upstream.uri());
        // Member's TPM cap (10) is below its per-call token cost (16).
        snap.models.insert(direct_model_entry_rl(
            "m-capped",
            "capped",
            "panel-upstream",
            serde_json::json!({ "tpm": 10 }),
        ));
        snap.models
            .insert(direct_model_entry("m-judge", "judge-m", "judge-upstream"));
        // Single-member panel → min_responses defaults to 1.
        snap.models.insert(ensemble_model_entry(
            "m-council",
            "council",
            &["capped"],
            "judge-m",
        ));
        snap.apikeys.insert(apikey_entry("sk-caller", &["council"]));
        let app = build_router(build_state(snap, hub));

        let make_req = || {
            let body = serde_json::json!({
                "model": "council",
                "messages": [{"role": "user", "content": "what is the best answer?"}]
            });
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", "Bearer sk-caller")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap()
        };
        // First request commits the member's 16 tokens to its model: TPM bucket.
        assert_eq!(run(app.clone(), make_req()).await.status(), StatusCode::OK);
        // Second request: the committed 16 now exceeds tpm:10 → member refused →
        // panel below min_responses → 502.
        assert_eq!(run(app, make_req()).await.status(), StatusCode::BAD_GATEWAY);
    }

    /// #620 (audit M-2): a panel member's reservation must RELEASE on a bridge
    /// ERROR (not just on commit). The member is capped at concurrency:1 and the
    /// upstream 503s on the first call: request 1's member fails (panel < min →
    /// 502), and its reservation must drop, freeing the slot. Request 2 (upstream
    /// 200) must then acquire the slot and succeed. A leaked slot would starve
    /// request 2 → 502.
    #[tokio::test]
    async fn ensemble_panel_member_reservation_released_on_bridge_error() {
        let upstream = MockServer::start().await;
        // First panel call fails (503), one time only.
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(503))
            .up_to_n_times(1)
            .with_priority(1)
            .mount(&upstream)
            .await;
        // Judge synthesis (only reached on the successful second request).
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(wiremock::matchers::body_string_contains("Answer 1:"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-judge",
                "model": "judge-upstream",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "synthesized final answer"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 30, "completion_tokens": 7, "total_tokens": 37}
            })))
            .with_priority(2)
            .mount(&upstream)
            .await;
        // Panel candidate (the second request's panel call, after the 503 is spent).
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-panel",
                "model": "panel-upstream",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "a panel candidate answer"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 5, "completion_tokens": 11, "total_tokens": 16}
            })))
            .with_priority(3)
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = new_snap(&upstream.uri());
        snap.models.insert(direct_model_entry_rl(
            "m-capped",
            "capped",
            "panel-upstream",
            serde_json::json!({ "concurrency": 1 }),
        ));
        snap.models
            .insert(direct_model_entry("m-judge", "judge-m", "judge-upstream"));
        snap.models.insert(ensemble_model_entry(
            "m-council",
            "council",
            &["capped"],
            "judge-m",
        ));
        snap.apikeys.insert(apikey_entry("sk-caller", &["council"]));
        let app = build_router(build_state(snap, hub));

        let make_req = || {
            let body = serde_json::json!({
                "model": "council",
                "messages": [{"role": "user", "content": "what is the best answer?"}]
            });
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", "Bearer sk-caller")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap()
        };
        // Request 1: the member's upstream 503s → member fails → panel below
        // min_responses → 502. Its concurrency:1 slot must be released on drop.
        assert_eq!(
            run(app.clone(), make_req()).await.status(),
            StatusCode::BAD_GATEWAY
        );
        // Request 2: upstream now 200. The slot is free (released on the prior
        // error), so the member succeeds and the ensemble synthesizes → 200.
        assert_eq!(run(app, make_req()).await.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn ensemble_fans_out_to_panel_and_returns_judge_synthesis() {
        let upstream = MockServer::start().await;
        // Judge synthesis call — its prompt embeds the neutrally-labeled
        // candidate answers ("Answer 1:"). Highest priority so it wins
        // over the catch-all panel mock for the judge request only.
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(wiremock::matchers::body_string_contains("Answer 1:"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-judge",
                "model": "judge-upstream",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "synthesized final answer"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 30, "completion_tokens": 7, "total_tokens": 37}
            })))
            .with_priority(1)
            .mount(&upstream)
            .await;
        // Panel members — catch-all for any chat call that isn't the judge.
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-panel",
                "model": "panel-upstream",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "a panel candidate answer"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 5, "completion_tokens": 11, "total_tokens": 16}
            })))
            .with_priority(2)
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));

        let snap = new_snap(&upstream.uri());
        snap.models
            .insert(direct_model_entry("m-panel-a", "panel-a", "panel-upstream"));
        snap.models
            .insert(direct_model_entry("m-panel-b", "panel-b", "panel-upstream"));
        snap.models
            .insert(direct_model_entry("m-judge", "judge-m", "judge-upstream"));
        snap.models.insert(ensemble_model_entry(
            "m-council",
            "council",
            &["panel-a", "panel-b"],
            "judge-m",
        ));
        snap.apikeys.insert(apikey_entry("sk-caller", &["council"]));
        let app = build_router(build_state(snap, hub));

        let body = serde_json::json!({
            "model": "council",
            "messages": [{"role": "user", "content": "what is the best answer?"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let v: serde_json::Value =
            serde_json::from_slice(&to_bytes(resp.into_body(), 65536).await.unwrap()).unwrap();
        // The client sees the JUDGE's synthesized answer, not a panel one.
        assert_eq!(v["object"], "chat.completion");
        assert_eq!(
            v["choices"][0]["message"]["content"],
            "synthesized final answer"
        );
        // Rendered under the requested ensemble model name — never the
        // judge's upstream model id (no provider/model leakage).
        assert_eq!(v["model"], "council");
        // #614: client-facing usage is the AGGREGATE of every panel member
        // plus the judge, not the judge's alone — so the caller sees the full
        // fan-out cost. Here: two panel members at total_tokens=16 each (32)
        // plus the judge's 37 = 69.
        assert_eq!(v["usage"]["total_tokens"], 69);
    }

    /// Tool-using requests can't be fanned out coherently across a panel,
    /// so the ensemble path rejects them with a 400 before any upstream
    /// call. `tools` is a flattened key in the request body.
    #[tokio::test]
    async fn ensemble_rejects_tool_requests_with_400() {
        let upstream = MockServer::start().await;
        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));

        let snap = new_snap(&upstream.uri());
        snap.models
            .insert(direct_model_entry("m-panel-a", "panel-a", "panel-upstream"));
        snap.models
            .insert(direct_model_entry("m-judge", "judge-m", "judge-upstream"));
        snap.models.insert(ensemble_model_entry(
            "m-council",
            "council",
            &["panel-a"],
            "judge-m",
        ));
        snap.apikeys.insert(apikey_entry("sk-caller", &["council"]));
        let app = build_router(build_state(snap, hub));

        let body = serde_json::json!({
            "model": "council",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{
                "type": "function",
                "function": {"name": "get_weather", "parameters": {}}
            }]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let v: serde_json::Value =
            serde_json::from_slice(&to_bytes(resp.into_body(), 4096).await.unwrap()).unwrap();
        assert_eq!(v["error"]["type"], "invalid_request_error");
    }

    /// Streaming ensemble (OPTION A): the panel runs non-streaming (buffered
    /// to synthesize) and ONLY the judge's tokens are streamed back as SSE.
    /// The client must receive the judge's synthesized content + a `[DONE]`
    /// sentinel, every chunk re-stamped under the requested ensemble model
    /// name ("council") — never an upstream id. Telemetry must emit one event
    /// per panel member + one judge event, all sharing the same request_id.
    #[tokio::test]
    async fn ensemble_streams_judge_synthesis() {
        use aisix_obs::UsageSink;
        let upstream = MockServer::start().await;
        // Judge synthesis call (matched by its "Answer 1:" prompt) streams its
        // answer back as SSE — this is the leg the gateway now streams to the
        // client. The terminal chunk carries the judge's usage block.
        let judge_sse = "\
data: {\"id\":\"cmpl-judge\",\"model\":\"judge-upstream\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"cmpl-judge\",\"model\":\"judge-upstream\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"synthesized final answer\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"cmpl-judge\",\"model\":\"judge-upstream\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":30,\"completion_tokens\":7,\"total_tokens\":37}}\n\n\
data: [DONE]\n\n";
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(wiremock::matchers::body_string_contains("Answer 1:"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(judge_sse),
            )
            .with_priority(1)
            .mount(&upstream)
            .await;
        // Panel members — catch-all, NON-streaming JSON (the executor buffers
        // these to build the judge prompt).
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-panel",
                "model": "panel-upstream",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "a panel candidate answer"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 5, "completion_tokens": 11, "total_tokens": 16}
            })))
            .with_priority(2)
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));

        let snap = new_snap(&upstream.uri());
        snap.models
            .insert(direct_model_entry("m-panel-a", "panel-a", "panel-upstream"));
        snap.models
            .insert(direct_model_entry("m-panel-b", "panel-b", "panel-upstream"));
        snap.models
            .insert(direct_model_entry("m-judge", "judge-m", "judge-upstream"));
        snap.models.insert(ensemble_model_entry(
            "m-council",
            "council",
            &["panel-a", "panel-b"],
            "judge-m",
        ));
        snap.apikeys.insert(apikey_entry("sk-caller", &["council"]));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let app = build_router(build_state(snap, hub).with_usage_sink(UsageSink::new(tx)));

        let body = serde_json::json!({
            "model": "council",
            "messages": [{"role": "user", "content": "what is the best answer?"}],
            "stream": true
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default()
            .contains("text/event-stream"));

        // Drain + decode the SSE body. Assert the judge's content streamed
        // through, a [DONE] sentinel terminates it, and every data chunk is
        // re-stamped under "council" (no upstream id leakage).
        let mut body_stream = resp.into_body().into_data_stream();
        let mut decoder = SseDecoder::new();
        let mut events = Vec::new();
        while let Some(chunk) = body_stream.next().await {
            events.extend(decoder.feed(chunk.unwrap().as_ref()));
        }
        assert!(events.contains(&SseEvent::Done), "missing [DONE] sentinel");
        let data: Vec<&str> = events
            .iter()
            .filter_map(|e| match e {
                SseEvent::Data(s) => Some(s.as_str()),
                _ => None,
            })
            .collect();
        let joined = data.join("\n");
        assert!(
            joined.contains("synthesized final answer"),
            "client must receive the judge's synthesized content; got: {joined}"
        );
        // Every chunk's `model` is the ensemble alias, never the judge upstream.
        for d in &data {
            let v: serde_json::Value = serde_json::from_str(d).unwrap();
            assert_eq!(
                v["model"], "council",
                "streamed chunk must be re-stamped under the ensemble model name"
            );
            assert_ne!(v["model"], "judge-upstream");
        }

        // Telemetry (emitted from on_complete once the stream is drained):
        // two panel events + one judge event, all sharing one request_id.
        let mut usage = Vec::new();
        while let Ok(Some(ev)) =
            tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv()).await
        {
            usage.push(ev);
            if usage.len() == 3 {
                break;
            }
        }
        assert_eq!(
            usage.len(),
            3,
            "expected 3 usage events (2 panel + 1 judge); got {}",
            usage.len()
        );
        let panel_events: Vec<_> = usage.iter().filter(|e| e.attempt_kind == "panel").collect();
        let judge_events: Vec<_> = usage.iter().filter(|e| e.attempt_kind == "judge").collect();
        assert_eq!(panel_events.len(), 2, "both panel members must emit");
        assert_eq!(judge_events.len(), 1, "the judge must emit one event");
        let rid = usage[0].request_id.clone();
        assert!(
            !rid.is_empty() && usage.iter().all(|e| e.request_id == rid),
            "all sub-call events share the request_id (trace key)"
        );
        // The judge event carries the streamed terminal-chunk usage.
        assert_eq!(judge_events[0].prompt_tokens, 30);
        assert_eq!(judge_events[0].completion_tokens, 7);
    }

    /// #614: a STREAMING ensemble's client-facing terminal usage frame (sent
    /// when the client asked for `stream_options.include_usage`) is the
    /// panel+judge AGGREGATE, not the judge's alone — matching the
    /// non-streaming path. Here: two panel members at total_tokens=16 each
    /// (prompt 5 / completion 11) + the judge's streamed 37 (prompt 30 /
    /// completion 7) ⇒ total 69, prompt 40, completion 29.
    #[tokio::test]
    async fn ensemble_streaming_usage_frame_is_panel_judge_aggregate() {
        let upstream = MockServer::start().await;
        let judge_sse = "\
data: {\"id\":\"cmpl-judge\",\"model\":\"judge-upstream\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"synthesized final answer\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"cmpl-judge\",\"model\":\"judge-upstream\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":30,\"completion_tokens\":7,\"total_tokens\":37}}\n\n\
data: [DONE]\n\n";
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(wiremock::matchers::body_string_contains("Answer 1:"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(judge_sse),
            )
            .with_priority(1)
            .mount(&upstream)
            .await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-panel",
                "model": "panel-upstream",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "a panel candidate answer"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 5, "completion_tokens": 11, "total_tokens": 16}
            })))
            .with_priority(2)
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = new_snap(&upstream.uri());
        snap.models
            .insert(direct_model_entry("m-panel-a", "panel-a", "panel-upstream"));
        snap.models
            .insert(direct_model_entry("m-panel-b", "panel-b", "panel-upstream"));
        snap.models
            .insert(direct_model_entry("m-judge", "judge-m", "judge-upstream"));
        snap.models.insert(ensemble_model_entry(
            "m-council",
            "council",
            &["panel-a", "panel-b"],
            "judge-m",
        ));
        snap.apikeys.insert(apikey_entry("sk-caller", &["council"]));
        let app = build_router(build_state(snap, hub));

        let body = serde_json::json!({
            "model": "council",
            "messages": [{"role": "user", "content": "what is the best answer?"}],
            "stream": true,
            "stream_options": {"include_usage": true}
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);

        let mut body_stream = resp.into_body().into_data_stream();
        let mut decoder = SseDecoder::new();
        let mut events = Vec::new();
        while let Some(chunk) = body_stream.next().await {
            events.extend(decoder.feed(chunk.unwrap().as_ref()));
        }
        // The client asked for usage, so exactly one terminal usage-bearing
        // frame must reach it — carrying the aggregate, re-stamped under the
        // ensemble alias (never the judge upstream id).
        // Collect ALL usage-bearing frames: the client must receive EXACTLY
        // one (the panel sum is folded once). More than one would mean the
        // base_usage fold ran per-chunk against a multi-emit judge (#617).
        let usage_frames: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                SseEvent::Data(s) => serde_json::from_str::<serde_json::Value>(s).ok(),
                _ => None,
            })
            .filter(|v| !v["usage"].is_null())
            .collect();
        assert_eq!(
            usage_frames.len(),
            1,
            "client must receive exactly one usage frame (panel sum folded once)"
        );
        let usage_frame = &usage_frames[0];
        assert_eq!(
            usage_frame["usage"]["total_tokens"], 69,
            "aggregate total = 2*16 panel + 37 judge"
        );
        assert_eq!(
            usage_frame["usage"]["prompt_tokens"], 40,
            "aggregate prompt = 5 + 5 + 30"
        );
        assert_eq!(
            usage_frame["usage"]["completion_tokens"], 29,
            "aggregate completion = 11 + 11 + 7"
        );
        assert_eq!(usage_frame["model"], "council");
    }

    /// Streaming ensemble, judge connect failure: the panel members all
    /// succeed (and are billed) but the judge upstream returns a hard error
    /// on the streaming connect. The panel's usage events must STILL fire —
    /// every panel member round-tripped an upstream, exactly like the
    /// non-streaming judge-failure path.
    #[tokio::test]
    async fn ensemble_streaming_judge_connect_failure_still_bills_panel() {
        use aisix_obs::UsageSink;
        let upstream = MockServer::start().await;
        // Judge synthesis call (matched by "Answer 1:") → 500 on connect. A
        // non-2xx upstream status surfaces as a connect-time error from
        // `chat_stream`, so the gateway never commits a 200 to the client.
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(wiremock::matchers::body_string_contains("Answer 1:"))
            .respond_with(ResponseTemplate::new(500).set_body_json(serde_json::json!({
                "error": {"message": "judge upstream exploded", "type": "server_error"}
            })))
            .with_priority(1)
            .mount(&upstream)
            .await;
        // Panel members → 200 (catch-all). Both survive, so min_responses is
        // met and the run proceeds to the (failing) judge connect.
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-panel",
                "model": "panel-upstream",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "a panel candidate answer"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 5, "completion_tokens": 11, "total_tokens": 16}
            })))
            .with_priority(2)
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = new_snap(&upstream.uri());
        seed_two_member_council(&snap);
        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let app = build_router(build_state(snap, hub).with_usage_sink(UsageSink::new(tx)));

        let body = serde_json::json!({
            "model": "council",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = run(app, req).await;
        // Judge 5xx collapses to 502 for the client (connect failed before any
        // SSE byte was committed).
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);

        // Both panel members already hit upstream and were billed, so their
        // usage events must still fire. The judge produced no response → no
        // judge event.
        let mut events = Vec::new();
        while let Ok(Some(ev)) =
            tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv()).await
        {
            events.push(ev);
        }
        let panel_events: Vec<_> = events
            .iter()
            .filter(|e| e.attempt_kind == "panel")
            .collect();
        assert_eq!(
            panel_events.len(),
            2,
            "both billed panel members must emit a usage event; got {} events total",
            events.len()
        );
        assert!(
            panel_events
                .iter()
                .all(|e| e.prompt_tokens == 5 && e.completion_tokens == 11),
            "panel events must carry the panel call's own tokens"
        );
        assert!(
            events.iter().all(|e| e.attempt_kind != "judge"),
            "the judge connect failed, so no judge usage event"
        );
    }

    /// Like `ensemble_model_entry` but with an explicit `min_responses`.
    fn ensemble_model_entry_min(
        id: &str,
        name: &str,
        panel: &[&str],
        judge: &str,
        min_responses: u32,
    ) -> ResourceEntry<Model> {
        let panel_json = panel
            .iter()
            .map(|m| format!(r#"{{"model":"{m}"}}"#))
            .collect::<Vec<_>>()
            .join(",");
        let cfg = format!(
            r#"{{
                "display_name": "{name}",
                "ensemble": {{
                    "panel": [{panel_json}],
                    "judge": {{"model": "{judge}"}},
                    "min_responses": {min_responses}
                }}
            }}"#
        );
        let model: Model = serde_json::from_str(&cfg).unwrap();
        ResourceEntry::new(id, model, 1)
    }

    /// Output-guardrail block path — the case FIX 1 (billing) targets. The
    /// judge's synthesized answer trips an output keyword guardrail. The
    /// client must get the content-filtered status, AND every per-sub-call
    /// usage event (panel members + judge) must still fire with
    /// `guardrail_blocked == true` — the panel tokens are already committed,
    /// so dropping these events would under-report panel usage to cp-api.
    #[tokio::test]
    async fn ensemble_output_block_still_emits_panel_and_judge_usage() {
        use aisix_obs::UsageSink;
        let upstream = MockServer::start().await;
        // Judge synthesis call returns content that the output guardrail
        // blocks ("secret-string"). Highest priority for the judge request.
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(wiremock::matchers::body_string_contains("Answer 1:"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-judge",
                "model": "judge-upstream",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "here is the secret-string"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 30, "completion_tokens": 7, "total_tokens": 37}
            })))
            .with_priority(1)
            .mount(&upstream)
            .await;
        // Panel members — catch-all, benign content.
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-panel",
                "model": "panel-upstream",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "a panel candidate answer"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 5, "completion_tokens": 11, "total_tokens": 16}
            })))
            .with_priority(2)
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));

        let snap = new_snap(&upstream.uri());
        snap.models
            .insert(direct_model_entry("m-panel-a", "panel-a", "panel-upstream"));
        snap.models
            .insert(direct_model_entry("m-panel-b", "panel-b", "panel-upstream"));
        snap.models
            .insert(direct_model_entry("m-judge", "judge-m", "judge-upstream"));
        snap.models.insert(ensemble_model_entry(
            "m-council",
            "council",
            &["panel-a", "panel-b"],
            "judge-m",
        ));
        snap.apikeys.insert(apikey_entry("sk-caller", &["council"]));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let state = build_state(snap, hub).with_usage_sink(UsageSink::new(tx));
        // Output keyword guardrail blocking the judge's synthesized answer.
        seed_guardrail(
            &state.snapshot,
            "g-ensemble-out",
            r#"{"name":"ens-out-guard","kind":"keyword","hook_point":"output","patterns":[{"kind":"literal","value":"secret-string"}]}"#,
        );
        let app = build_router(state);

        let body = serde_json::json!({
            "model": "council",
            "messages": [{"role": "user", "content": "what is the answer?"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = run(app, req).await;
        // Content-filtered status reaches the client (ProxyError::ContentFiltered).
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let v: serde_json::Value =
            serde_json::from_slice(&to_bytes(resp.into_body(), 4096).await.unwrap()).unwrap();
        assert_eq!(v["error"]["type"], "content_filter");

        // Drain the usage sink: the two panel members + the judge must all
        // have emitted, each flagged guardrail_blocked (FIX 1 — the panel
        // bill is not lost on a block).
        let mut events = Vec::new();
        while let Ok(Some(ev)) =
            tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv()).await
        {
            events.push(ev);
            if events.len() == 3 {
                break;
            }
        }
        assert_eq!(
            events.len(),
            3,
            "expected 3 usage events (2 panel + judge) on the block path; got {}",
            events.len()
        );
        let panel_events: Vec<_> = events
            .iter()
            .filter(|e| e.attempt_kind == "panel")
            .collect();
        let judge_events: Vec<_> = events
            .iter()
            .filter(|e| e.attempt_kind == "judge")
            .collect();
        assert_eq!(panel_events.len(), 2, "both panel members must emit");
        assert_eq!(judge_events.len(), 1, "the judge must emit");
        assert!(
            events.iter().all(|e| e.guardrail_blocked),
            "every sub-call event on the block path must be guardrail_blocked"
        );
        // Panel token counts survive (the bug under-reported these).
        assert!(
            panel_events
                .iter()
                .all(|e| e.prompt_tokens == 5 && e.completion_tokens == 11),
            "panel usage must carry the panel call's own tokens"
        );
        assert_eq!(judge_events[0].prompt_tokens, 30);
        assert_eq!(judge_events[0].completion_tokens, 7);
    }

    /// A panel that can't reach `min_responses` (one member 503s, min=2 on a
    /// 2-member panel) surfaces as a 502 to the client — the executor's
    /// `InsufficientPanel` maps to an upstream-fault status. The SURVIVING
    /// member already hit upstream and was billed, so its usage event must
    /// still fire (FIX C — billed panel work is not lost on the 502 path).
    #[tokio::test]
    async fn ensemble_insufficient_panel_returns_502() {
        use aisix_obs::UsageSink;
        let upstream = MockServer::start().await;
        // The failing panel member (distinct upstream model name) → 503.
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(wiremock::matchers::body_string_contains(
                "panel-fail-upstream",
            ))
            .respond_with(ResponseTemplate::new(503).set_body_json(serde_json::json!({
                "error": {"message": "upstream busy", "type": "server_error"}
            })))
            .with_priority(1)
            .mount(&upstream)
            .await;
        // Everything else (the surviving panel member) → 200. The judge is
        // never reached because min_responses isn't met.
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-panel",
                "model": "panel-ok-upstream",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "only survivor"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 5, "completion_tokens": 11, "total_tokens": 16}
            })))
            .with_priority(2)
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));

        let snap = new_snap(&upstream.uri());
        snap.models.insert(direct_model_entry(
            "m-panel-ok",
            "panel-ok",
            "panel-ok-upstream",
        ));
        snap.models.insert(direct_model_entry(
            "m-panel-fail",
            "panel-fail",
            "panel-fail-upstream",
        ));
        snap.models
            .insert(direct_model_entry("m-judge", "judge-m", "judge-upstream"));
        snap.models.insert(ensemble_model_entry_min(
            "m-council",
            "council",
            &["panel-ok", "panel-fail"],
            "judge-m",
            2,
        ));
        snap.apikeys.insert(apikey_entry("sk-caller", &["council"]));
        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let app = build_router(build_state(snap, hub).with_usage_sink(UsageSink::new(tx)));

        let body = serde_json::json!({
            "model": "council",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);

        // The surviving panel member's usage event must still fire (FIX C):
        // it hit upstream and was billed even though the request 502'd. The
        // failed member and the never-run judge emit nothing here.
        let mut events = Vec::new();
        while let Ok(Some(ev)) =
            tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv()).await
        {
            events.push(ev);
        }
        let panel_events: Vec<_> = events
            .iter()
            .filter(|e| e.attempt_kind == "panel")
            .collect();
        assert_eq!(
            panel_events.len(),
            1,
            "exactly the one surviving panel member must emit a usage event; got {} events total",
            events.len()
        );
        assert_eq!(panel_events[0].prompt_tokens, 5);
        assert_eq!(panel_events[0].completion_tokens, 11);
        assert!(
            !panel_events[0].guardrail_blocked,
            "the survivor's event is a normal (non-blocked) bill"
        );
        assert!(
            events.iter().all(|e| e.attempt_kind != "judge"),
            "the judge never ran, so no judge usage event"
        );
    }

    /// Mount the standard panel + judge upstreams (judge matched by its
    /// "Answer 1:" synthesis prompt; everything else is a panel member).
    async fn mount_panel_and_judge(upstream: &MockServer) {
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(wiremock::matchers::body_string_contains("Answer 1:"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-judge",
                "model": "judge-upstream",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "synthesized final answer"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 30, "completion_tokens": 7, "total_tokens": 37}
            })))
            .with_priority(1)
            .mount(upstream)
            .await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-panel",
                "model": "panel-upstream",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "a panel candidate answer"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 5, "completion_tokens": 11, "total_tokens": 16}
            })))
            .with_priority(2)
            .mount(upstream)
            .await;
    }

    /// Seed a 2-member council (panel-a, panel-b → judge-m) + the api key.
    fn seed_two_member_council(snap: &AisixSnapshot) {
        snap.models
            .insert(direct_model_entry("m-panel-a", "panel-a", "panel-upstream"));
        snap.models
            .insert(direct_model_entry("m-panel-b", "panel-b", "panel-upstream"));
        snap.models
            .insert(direct_model_entry("m-judge", "judge-m", "judge-upstream"));
        snap.models.insert(ensemble_model_entry(
            "m-council",
            "council",
            &["panel-a", "panel-b"],
            "judge-m",
        ));
        snap.apikeys.insert(apikey_entry("sk-caller", &["council"]));
    }

    /// FIX D: an empty `tools: []` (which many SDKs always send) means "no
    /// tools" and must still fan out — NOT a 400.
    #[tokio::test]
    async fn ensemble_allows_empty_tools_array() {
        let upstream = MockServer::start().await;
        mount_panel_and_judge(&upstream).await;
        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = new_snap(&upstream.uri());
        seed_two_member_council(&snap);
        let app = build_router(build_state(snap, hub));

        let body = serde_json::json!({
            "model": "council",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": []
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "empty tools:[] must fan out, not 400"
        );
        let v: serde_json::Value =
            serde_json::from_slice(&to_bytes(resp.into_body(), 65536).await.unwrap()).unwrap();
        assert_eq!(
            v["choices"][0]["message"]["content"],
            "synthesized final answer"
        );
    }

    /// FIX D: `tool_choice: "none"` does not force a tool call, so it must
    /// still fan out.
    #[tokio::test]
    async fn ensemble_allows_tool_choice_none() {
        let upstream = MockServer::start().await;
        mount_panel_and_judge(&upstream).await;
        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = new_snap(&upstream.uri());
        seed_two_member_council(&snap);
        let app = build_router(build_state(snap, hub));

        let body = serde_json::json!({
            "model": "council",
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": "none"
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "tool_choice:\"none\" must fan out, not 400"
        );
    }

    /// FIX D: a forcing `tool_choice` (object form selecting a function) still
    /// 400s — the ensemble can't honour a forced tool call.
    #[tokio::test]
    async fn ensemble_rejects_forcing_tool_choice_with_400() {
        let upstream = MockServer::start().await;
        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = new_snap(&upstream.uri());
        seed_two_member_council(&snap);
        let app = build_router(build_state(snap, hub));

        let body = serde_json::json!({
            "model": "council",
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}}
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let v: serde_json::Value =
            serde_json::from_slice(&to_bytes(resp.into_body(), 4096).await.unwrap()).unwrap();
        assert_eq!(v["error"]["type"], "invalid_request_error");
    }

    /// FIX A leak probe: a misconfigured JUDGE (its `provider_key_id` points
    /// at a non-existent PK) surfaces an error to the client, but the
    /// envelope `error.message` must contain NEITHER the judge display_name
    /// NOR the dangling pk id — both are operator-internal config.
    #[tokio::test]
    async fn ensemble_misconfigured_judge_does_not_leak_internal_config() {
        let upstream = MockServer::start().await;
        // Panel members succeed so the run reaches the judge.
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-panel",
                "model": "panel-upstream",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "a panel candidate answer"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 5, "completion_tokens": 11, "total_tokens": 16}
            })))
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));

        let snap = new_snap(&upstream.uri());
        snap.models
            .insert(direct_model_entry("m-panel-a", "panel-a", "panel-upstream"));
        snap.models
            .insert(direct_model_entry("m-panel-b", "panel-b", "panel-upstream"));
        // The judge references a provider_key_id that is NOT in the snapshot.
        let secret_judge_name = "secret-judge-name";
        let dangling_pk = "99999999-9999-9999-9999-999999999999";
        let judge_cfg = format!(
            r#"{{"display_name":"{secret_judge_name}","provider":"openai","model_name":"judge-upstream","provider_key_id":"{dangling_pk}"}}"#
        );
        let judge_model: Model = serde_json::from_str(&judge_cfg).unwrap();
        snap.models
            .insert(ResourceEntry::new("m-judge-bad", judge_model, 1));
        snap.models.insert(ensemble_model_entry(
            "m-council",
            "council",
            &["panel-a", "panel-b"],
            secret_judge_name,
        ));
        snap.apikeys.insert(apikey_entry("sk-caller", &["council"]));
        let app = build_router(build_state(snap, hub));

        let body = serde_json::json!({
            "model": "council",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = run(app, req).await;
        // The judge can't be dispatched → an error reaches the client.
        assert!(
            resp.status().is_client_error() || resp.status().is_server_error(),
            "misconfigured judge must surface an error; got {}",
            resp.status()
        );
        let v: serde_json::Value =
            serde_json::from_slice(&to_bytes(resp.into_body(), 4096).await.unwrap()).unwrap();
        let message = v["error"]["message"].as_str().unwrap_or_default();
        assert!(
            !message.contains(secret_judge_name),
            "envelope must not leak the judge display_name; got: {message:?}"
        );
        assert!(
            !message.contains(dangling_pk),
            "envelope must not leak the provider_key_id; got: {message:?}"
        );
    }

    /// FIX E: a non-chat endpoint must reject an ensemble model with an
    /// explicit, accurate message (not the misleading "routing models"
    /// branch). /v1/embeddings is the probe.
    #[tokio::test]
    async fn ensemble_model_on_embeddings_returns_400_with_explicit_message() {
        let upstream = MockServer::start().await;
        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = new_snap(&upstream.uri());
        snap.models
            .insert(direct_model_entry("m-panel-a", "panel-a", "panel-upstream"));
        snap.models
            .insert(direct_model_entry("m-judge", "judge-m", "judge-upstream"));
        snap.models.insert(ensemble_model_entry(
            "m-council",
            "council",
            &["panel-a"],
            "judge-m",
        ));
        snap.apikeys.insert(apikey_entry("sk-caller", &["council"]));
        let app = build_router(build_state(snap, hub));

        let body = serde_json::json!({
            "model": "council",
            "input": "embed me"
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/embeddings")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let v: serde_json::Value =
            serde_json::from_slice(&to_bytes(resp.into_body(), 4096).await.unwrap()).unwrap();
        assert_eq!(v["error"]["type"], "invalid_request_error");
        let message = v["error"]["message"].as_str().unwrap_or_default();
        assert!(
            message.contains("ensemble model") && message.contains("/v1/chat/completions"),
            "non-chat endpoint must name the ensemble + chat-only constraint; got: {message:?}"
        );
        // Must NOT mislead by calling it a routing model.
        assert!(
            !message.contains("routing"),
            "ensemble rejection must not say 'routing'; got: {message:?}"
        );
    }

    /// FIX G: the full panel succeeds and is billed, then the judge upstream
    /// 500s. The client gets a 502 (judge 5xx collapses), and the panel
    /// members' usage events must STILL fire (they hit upstream) — with zero
    /// judge events, since the judge produced no response.
    #[tokio::test]
    async fn ensemble_judge_failure_still_bills_panel() {
        use aisix_obs::UsageSink;
        let upstream = MockServer::start().await;
        // Judge synthesis call (matched by its "Answer 1:" prompt) → 500.
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(wiremock::matchers::body_string_contains("Answer 1:"))
            .respond_with(ResponseTemplate::new(500).set_body_json(serde_json::json!({
                "error": {"message": "judge upstream exploded", "type": "server_error"}
            })))
            .with_priority(1)
            .mount(&upstream)
            .await;
        // Panel members → 200 (catch-all). Both survive, so min_responses is
        // met and the run proceeds to the (failing) judge.
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-panel",
                "model": "panel-upstream",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "a panel candidate answer"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 5, "completion_tokens": 11, "total_tokens": 16}
            })))
            .with_priority(2)
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = new_snap(&upstream.uri());
        seed_two_member_council(&snap);
        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let app = build_router(build_state(snap, hub).with_usage_sink(UsageSink::new(tx)));

        let body = serde_json::json!({
            "model": "council",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = run(app, req).await;
        // Judge 5xx collapses to 502 for the client.
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);

        // Both panel members already hit upstream and were billed, so their
        // usage events must still fire (FIX G). The judge produced no
        // response → no judge event.
        let mut events = Vec::new();
        while let Ok(Some(ev)) =
            tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv()).await
        {
            events.push(ev);
        }
        let panel_events: Vec<_> = events
            .iter()
            .filter(|e| e.attempt_kind == "panel")
            .collect();
        assert_eq!(
            panel_events.len(),
            2,
            "both billed panel members must emit a usage event; got {} events total",
            events.len()
        );
        assert!(
            panel_events
                .iter()
                .all(|e| e.prompt_tokens == 5 && e.completion_tokens == 11),
            "panel events must carry the panel call's own tokens"
        );
        assert!(
            panel_events.iter().all(|e| !e.guardrail_blocked),
            "the judge-failure panel bill is not a guardrail block"
        );
        assert!(
            events.iter().all(|e| e.attempt_kind != "judge"),
            "the judge produced no response, so no judge usage event"
        );
    }
}
