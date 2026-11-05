//! The [`Bridge`] trait — what every provider crate implements.
//!
//! A Bridge is the provider-specific adapter between the gateway's
//! normalised [`ChatFormat`] and whichever upstream API shape the vendor
//! requires. Bridges are held in [`crate::hub::Hub`] and selected by the
//! Model's [`aisix_core::Provider`] enum.
//!
//! Responsibilities of a Bridge:
//! - Translate `ChatFormat` → upstream request body
//! - Perform the HTTP call (authorisation, timeouts, retries at transport)
//! - For streaming requests, produce a `Stream<Item = ChatChunk>`
//! - For non-streaming, produce a full [`ChatResponse`]
//! - Surface errors as typed [`BridgeError`] variants so the proxy layer
//!   can map them to consistent OpenAI-style error envelopes
//!
//! The trait is deliberately `async_trait` rather than GATs — ergonomic
//! wins outweigh the boxing cost on the provider path.

use aisix_core::{HeaderVars, Model, ProviderKey};
use async_trait::async_trait;
use futures::stream::BoxStream;
use http::HeaderMap;
use std::time::Duration;

use crate::chat::{ChatChunk, ChatFormat, ChatResponse, EmbeddingRequest, EmbeddingResponse};
use crate::upstream_headers::{CallerIdentity, UpstreamHeaderContext};

/// Maximum number of bytes read from an upstream error response body
/// before attempting JSON envelope parse. Bounds memory and parser cost
/// when an upstream returns something pathological (an HTML error page
/// from a fronting WAF, or an unexpectedly large debug dump).
pub const MAX_UPSTREAM_ERROR_BODY_BYTES: usize = 64 * 1024;

/// Maximum length of the human-readable `message` string carried inside
/// [`BridgeError::UpstreamStatus`]. The full body is parsed into
/// [`UpstreamErrorView`] when JSON-shaped; the truncated string is the
/// fallback shown to clients when parsing fails.
pub const MAX_UPSTREAM_ERROR_MESSAGE_BYTES: usize = 1024;

/// Which wire format the upstream that produced this error speaks. The
/// envelope-rendering layer uses this together with [`UpstreamErrorView`]
/// to decide whether the upstream `kind` / `code` can be forwarded
/// verbatim or needs translation to the client's wire shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpstreamWire {
    /// OpenAI-compatible envelope: `{error:{message,type,code,param}}`.
    OpenAI,
    /// Anthropic envelope: `{type:"error",error:{type,message}}`.
    Anthropic,
    /// Azure OpenAI envelope: OpenAI-like with `error.inner_error.code`
    /// quirks for content policy violations.
    AzureOpenAI,
    /// AWS Bedrock structured error from the strongly-typed SDK; `kind`
    /// carries the AWS exception code (e.g. `"ThrottlingException"`).
    Bedrock,
    /// Vertex AI envelope: `{error:{code:int,message,status}}` where
    /// `status` is the canonical gRPC code string.
    Vertex,
    /// Wire format unknown / not applicable (tests, synthesised errors,
    /// the legacy convenience constructors). Renders as the generic
    /// `upstream_error` envelope with no translation attempt.
    Unknown,
}

/// Structured view of an upstream error envelope, populated by each
/// bridge after best-effort parsing of its provider's known shape.
/// `None` everywhere means parsing failed (non-JSON body, malformed
/// JSON, or unfamiliar envelope shape); callers fall back to the
/// truncated raw message on [`BridgeError::UpstreamStatus::message`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct UpstreamErrorView {
    /// Provider-native error-type token, unchanged from the upstream
    /// envelope (e.g. Anthropic `"rate_limit_error"`, OpenAI
    /// `"rate_limit_exceeded"`, Bedrock `"ThrottlingException"`).
    pub kind: Option<String>,
    /// Human-readable upstream message, post-parse.
    pub message: Option<String>,
    /// OpenAI envelope only. Other providers populate via the
    /// translation table at render time.
    pub code: Option<String>,
    /// OpenAI envelope only.
    pub param: Option<String>,
}

/// Context carried through the whole request lifecycle.
///
/// The proxy layer fills this in after it has authenticated the request
/// and resolved both the target Model AND its referenced ProviderKey
/// from the [`aisix_core::AisixSnapshot`]. Bridges read from it but
/// do not mutate it.
#[derive(Clone)]
pub struct BridgeContext {
    /// Correlation id propagated into traces and error envelopes.
    pub request_id: String,
    /// The resolved Model — bridges read `model_name` (the upstream
    /// model id) and metadata (timeout, rate_limit) from here.
    pub model: std::sync::Arc<Model>,
    /// The ProviderKey the Model references — bridges read `secret`
    /// (api key) and `api_base` (optional override) from here.
    pub provider_key: std::sync::Arc<ProviderKey>,
    /// Deadline for the entire upstream call. Bridges are expected to
    /// honour this by cancelling any in-flight HTTP request.
    ///
    /// On a streaming dispatch this is the **streaming** budget, which
    /// bounds the connect phase and the gap between chunks rather than
    /// the whole completion. A bridge that answers a streaming request
    /// with a non-streaming upstream leg must use
    /// [`non_streaming_deadline`](Self::non_streaming_deadline) instead.
    pub deadline: Option<Duration>,
    /// The end-to-end budget for a non-streaming upstream call, carried
    /// alongside `deadline` on streaming dispatches.
    ///
    /// A structured-output request on the synthetic-tool route cannot be
    /// streamed — the JSON only exists once the tool call is complete —
    /// so those bridges run the upstream leg non-streaming and render
    /// the result as chunks. Measured against the streaming budget, a
    /// completion that takes longer than one chunk gap is supposed to
    /// would be cut off; this is the budget that call is actually
    /// entitled to. `None` on a non-streaming dispatch, where `deadline`
    /// already is it.
    pub non_streaming_deadline: Option<Duration>,
    /// The authenticated caller, for `${request.api_key.*}` header
    /// templates. Default (all-empty) on calls with no caller behind
    /// them — a background job poll, an internal embedding lookup.
    pub caller: CallerIdentity,
    /// The inbound request's headers, source for the ProviderKey's
    /// `request.forward_client_headers` allowlist. `None` on the same
    /// caller-less paths as above.
    pub client_headers: Option<std::sync::Arc<HeaderMap>>,
    /// Snapshot ids of the resolved Model and ProviderKey, for the
    /// `${model.id}` / `${provider_key.id}` header templates. They are
    /// carried separately because a `Model` / `ProviderKey` value does not
    /// know its own etcd id at runtime — the id lives on the enclosing
    /// `ResourceEntry`, and `Resource::id()` on the value reads an
    /// unpopulated field outside tests.
    pub model_id: String,
    pub provider_key_id: String,
    /// Size of the caller's inbound request body, which media the gateway
    /// downloads for the request counts against together with
    /// `proxy.request_body_limit_bytes`. `0` on caller-less calls.
    pub inbound_body_bytes: usize,
}

/// Print the context without its credentials.
///
/// Two of these fields reach live secrets: `provider_key` carries the
/// gateway's own upstream `api_key`, and `client_headers` the caller's
/// own. A derived `Debug` would put both in any log line that ever
/// formats a context, so the type names them instead of quoting them.
impl std::fmt::Debug for BridgeContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BridgeContext")
            .field("request_id", &self.request_id)
            .field("model", &self.model.display_name)
            .field("provider_key", &self.provider_key.display_name)
            .field("deadline", &self.deadline)
            .field("caller", &self.caller)
            // Names only: the inbound map holds the caller's own
            // `Authorization`, and nothing marks a header sensitive on
            // the way in.
            .field(
                "client_headers",
                &self
                    .client_headers
                    .as_ref()
                    .map(|h| h.keys().map(|k| k.as_str()).collect::<Vec<_>>()),
            )
            .field("model_id", &self.model_id)
            .field("provider_key_id", &self.provider_key_id)
            .finish()
    }
}

impl BridgeContext {
    pub fn new(
        request_id: impl Into<String>,
        model: std::sync::Arc<Model>,
        provider_key: std::sync::Arc<ProviderKey>,
    ) -> Self {
        Self {
            request_id: request_id.into(),
            model,
            provider_key,
            deadline: None,
            non_streaming_deadline: None,
            caller: CallerIdentity::default(),
            client_headers: None,
            model_id: String::new(),
            provider_key_id: String::new(),
            inbound_body_bytes: 0,
        }
    }

    pub fn with_inbound_body_bytes(mut self, bytes: usize) -> Self {
        self.inbound_body_bytes = bytes;
        self
    }

    pub fn with_deadline(mut self, deadline: Duration) -> Self {
        self.deadline = Some(deadline);
        self
    }

    /// Record the end-to-end budget a non-streaming call would have got,
    /// for the streaming dispatches whose `deadline` is the smaller
    /// streaming budget. See
    /// [`non_streaming_deadline`](Self::non_streaming_deadline).
    pub fn with_non_streaming_deadline(mut self, deadline: Option<Duration>) -> Self {
        self.non_streaming_deadline = deadline;
        self
    }

    /// The deadline an upstream leg that is *not* streaming should run
    /// under, whichever kind of dispatch this context came from.
    pub fn non_streaming_ctx(&self) -> Self {
        let mut ctx = self.clone();
        // The result is a non-streaming context, which carries its budget
        // in `deadline` alone.
        if let Some(deadline) = ctx.non_streaming_deadline.take() {
            ctx.deadline = Some(deadline);
        }
        ctx
    }

    /// Attach the caller identity and inbound headers the outbound-header
    /// pipeline reads. Dispatch paths with a real client request call this;
    /// leaving it off means no client header is ever forwarded and
    /// `${request.api_key.*}` templates do not resolve.
    pub fn with_client(
        mut self,
        caller: CallerIdentity,
        client_headers: Option<std::sync::Arc<HeaderMap>>,
    ) -> Self {
        self.caller = caller;
        self.client_headers = client_headers;
        self
    }

    /// Attach the snapshot ids of the resolved Model / ProviderKey. See
    /// the field docs for why they cannot be read off the values.
    pub fn with_resource_ids(
        mut self,
        model_id: impl Into<String>,
        provider_key_id: impl Into<String>,
    ) -> Self {
        self.model_id = model_id.into();
        self.provider_key_id = provider_key_id.into();
        self
    }

    /// The context [`crate::upstream_headers::apply_request_headers`] needs
    /// to render `default_headers` templates and forward client headers.
    pub fn header_ctx(&self) -> UpstreamHeaderContext<'_> {
        UpstreamHeaderContext {
            overrides: self.provider_key.request.as_ref(),
            vars: HeaderVars {
                request_id: Some(&self.request_id),
                api_key_id: Some(&self.caller.api_key_id),
                api_key_name: self.caller.api_key_name.as_deref(),
                api_key_team_id: self.caller.team_id.as_deref(),
                api_key_user_id: self.caller.user_id.as_deref(),
                model_id: Some(&self.model_id),
                model_name: Some(&self.model.display_name),
                provider_key_id: Some(&self.provider_key_id),
                provider_key_name: Some(&self.provider_key.display_name),
            },
            client_headers: self.client_headers.as_deref(),
            // A Bridge speaks HTTP to its upstream and owns no header
            // name the shared lists do not already cover.
            surface_blocked: &[],
        }
    }
}

/// `": {cause}"` when a transport-layer cause is known, otherwise empty —
/// keeps the timeout message unchanged for the gateway's own deadlines.
fn timeout_cause_suffix(cause: &str) -> String {
    if cause.is_empty() {
        String::new()
    } else {
        format!(": {cause}")
    }
}

/// A non-chat operation a Bridge may or may not implement.
///
/// The value domain is closed on purpose: it names exactly the three
/// [`Bridge`] methods that ship a default implementation, and it is what
/// [`BridgeError::UnsupportedCapability`] carries. Before it existed the
/// proxy decided the 501 by searching the error's message text for a
/// phrase, so rewording the sentence below would have silently turned a
/// 501 into a 500 with nothing to catch it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BridgeCapability {
    /// [`Bridge::embed`] — `/v1/embeddings`.
    Embeddings,
    /// [`Bridge::complete`] — `/v1/completions`.
    TextCompletions,
    /// [`Bridge::generate_image`] — `/v1/images/generations`.
    ImageGeneration,
    /// [`Bridge::anthropic_messages`] and [`Bridge::anthropic_count_tokens`]
    /// — `/v1/messages` served on the platform's own Claude routes.
    AnthropicMessages,
}

impl BridgeCapability {
    /// The words the client-facing message names this capability by.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Embeddings => "embeddings",
            Self::TextCompletions => "text completions",
            Self::ImageGeneration => "image generation",
            Self::AnthropicMessages => "Anthropic messages",
        }
    }
}

impl std::fmt::Display for BridgeCapability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Error surfaced by any Bridge. Each variant maps to a stable
/// client-visible HTTP status and OpenAI-style error code so the proxy
/// layer can translate without further inspection.
#[derive(Debug, thiserror::Error)]
pub enum BridgeError {
    /// An upstream call exceeded a time budget.
    ///
    /// `cause` names the transport-layer reason when reqwest reported one,
    /// and is empty when one of the gateway's own deadlines elapsed. Three
    /// unrelated conditions all satisfy `reqwest::Error::is_timeout()` — a
    /// `connect_timeout`, hyper's request timeout, and the kernel's own
    /// `ETIMEDOUT` on an unanswered SYN — so without the cause they render
    /// as the same sentence and an operator cannot tell "the upstream is
    /// slow" from "we never reached it" (AISIX-Cloud#1093).
    #[error("upstream request timed out after {elapsed_ms}ms{}", timeout_cause_suffix(.cause))]
    Timeout { elapsed_ms: u64, cause: String },
    /// Upstream returned a non-2xx HTTP status. `retry_after` carries
    /// the upstream's `Retry-After` header parsed to a Duration when
    /// present — used by the cooldown layer to honor provider-supplied
    /// backoff hints. Bridges that cannot parse the header (or where
    /// the header is absent) leave this `None`; the cooldown layer
    /// falls back to its configured default in that case.
    /// `message` is a best-effort human-readable string for logs and
    /// the fallback envelope when [`parsed`] is `None`. When [`parsed`]
    /// is `Some`, the envelope-rendering layer (`error_translate`) uses
    /// the structured fields and [`wire`] to produce a client-shape
    /// envelope; `message` is kept around for logs and as a
    /// last-resort fallback if a parsed field is missing.
    ///
    /// `message` also decides the client's envelope (blank falls back to the
    /// status and reason phrase), so what logs and telemetry say beyond it
    /// lives in the error's `Display` and never in this field.
    #[error("{}", upstream_status_display(*.status, .message, .diagnostic.as_deref()))]
    UpstreamStatus {
        status: u16,
        message: String,
        /// Boxed to keep [`BridgeError`] small enough that
        /// `Result<_, ProxyError>` doesn't trip `clippy::result_large_err`
        /// once the four optional envelope fields are added.
        parsed: Option<Box<UpstreamErrorView>>,
        wire: UpstreamWire,
        retry_after: Option<Duration>,
        /// What `Display` says instead of the `message` wording, when the
        /// caller's text is not the diagnosis. See
        /// [`with_body_read_error`](Self::with_body_read_error) and
        /// [`with_telemetry_detail`](Self::with_telemetry_detail).
        diagnostic: Option<String>,
    },
    /// A non-2xx whose body could not be read, on a route that reports that
    /// to its caller as [`UpstreamDecode`](Self::UpstreamDecode): the same
    /// status, type, class and retry decisions, the same text to the caller
    /// (`decode_error`), and only `Display` saying what happened.
    #[error("upstream returned HTTP {status}: failed to read response body: {reason}")]
    UpstreamBodyUnread {
        status: u16,
        decode_error: String,
        reason: String,
    },
    /// A 2xx model answer the gateway could not read. Never retried,
    /// failed over or counted toward cooldown: the upstream already served
    /// — and billed — the call.
    #[error("upstream returned an unparseable body: {0}")]
    UpstreamDecode(String),
    /// A 2xx from a credential token exchange (Azure AAD, Vertex service
    /// account) the gateway could not read. Nothing was served or billed,
    /// so unlike [`UpstreamDecode`](Self::UpstreamDecode) it is retried and
    /// counts toward cooldown like a transport fault.
    #[error("upstream returned an unparseable body: {0}")]
    TokenExchangeDecode(String),
    /// A provider-reported error carried *inside* a 2xx streaming
    /// response body — an OpenAI-family `data: {"error":{...}}` frame,
    /// an Anthropic `event: error` frame, a Gemini in-stream error
    /// object, or a Bedrock event-stream modeled exception. By the time
    /// one arrives the upstream HTTP status was already a success, so
    /// [`UpstreamStatus`](Self::UpstreamStatus) cannot represent it.
    /// `status` is the numeric code embedded in (or documented for) the
    /// error body when the provider supplies one; `None` when the
    /// envelope carries no numeric code. Distinguishing this from
    /// [`UpstreamDecode`](Self::UpstreamDecode) keeps the provider's
    /// own error type/message intact instead of surfacing a serde
    /// parse failure (AISIX-Cloud#1222 scenario 3).
    #[error("{}", in_band_display(.message, .parsed.as_deref()))]
    UpstreamInBand {
        status: Option<u16>,
        message: String,
        /// Boxed for the same `result_large_err` reason as
        /// [`UpstreamStatus`](Self::UpstreamStatus).
        parsed: Option<Box<UpstreamErrorView>>,
        wire: UpstreamWire,
    },
    #[error("bridge is misconfigured: {0}")]
    Config(String),
    /// The provider's bridge does not implement this operation at all —
    /// the [`Bridge`] default impl answered, so no request was built and no
    /// upstream was contacted.
    ///
    /// Static per adapter: the same provider will answer the same way for
    /// every request, which is why it is the one failure that must not
    /// spend a retry ([`crate::BridgeError`] consumers in the proxy's
    /// `routing::is_retryable`).
    ///
    /// It carries the SAME status, error type and telemetry class as
    /// [`Config`](Self::Config), which it was spelled as until #1093 — the
    /// point of the variant is that a status decision no longer reads the
    /// message text, not that any response changed.
    #[error("this provider does not support {0}")]
    UnsupportedCapability(BridgeCapability),
    /// Customer-fixable upstream config — the admin's ProviderKey/Model
    /// is set up wrong (missing api_base, missing model_name). A request
    /// shape the bridge cannot translate is [`InvalidRequest`] instead,
    /// whose message does not blame the upstream config. Maps to
    /// 400, not 500: it's the caller's mistake, retrying won't help, and
    /// a 5xx wrongly tells SDKs/monitoring it's a server fault (#367).
    /// Contrast [`Config`], reserved for errors *we* cause
    /// (serialization, our generated request_id) which stays 500.
    #[error("invalid upstream configuration: {0}")]
    InvalidUpstreamConfig(String),
    /// The caller's request asks for something this provider's bridge
    /// cannot translate (a request field, tool type, or history shape).
    /// Same 400 `invalid_request_error` and non-retryable handling as
    /// [`InvalidUpstreamConfig`], but the message is the sentence itself
    /// rather than one prefixed as an upstream misconfiguration, so the
    /// caller is pointed at the request instead of the ProviderKey/Model,
    /// and the telemetry class is `invalid_request_error`, the one every
    /// refusal of the caller's request records, not `invalid_config`.
    #[error("{0}")]
    InvalidRequest(String),
    /// The request outgrew `proxy.request_body_limit_bytes` once the media
    /// the gateway downloaded for it is counted with its inbound body.
    /// The same 413 an over-limit inbound body gets; never retried, and
    /// raised before the upstream is contacted. The message counts both
    /// parts, since the inbound body alone is under the limit.
    #[error(
        "the request body ({inbound_bytes} bytes) plus downloaded media ({downloaded_bytes} \
         bytes) exceeds the {limit_bytes}-byte request_body_limit_bytes limit"
    )]
    RequestTooLarge {
        limit_bytes: usize,
        inbound_bytes: usize,
        downloaded_bytes: u64,
    },
    /// Customer-fixable upstream *credential* problem — the admin's
    /// ProviderKey secret/credential is missing, empty, or malformed
    /// (empty secret, api key with invalid HTTP-header bytes, unparseable
    /// service-account / AAD / Bedrock credential JSON). Maps to 401
    /// `authentication_error`, not 400: this is an auth-material problem,
    /// not a request/routing-shape problem. Non-retryable
    /// (#367 follow-up). Distinct from [`InvalidUpstreamConfig`] (400),
    /// which is request/routing shape, not credentials.
    #[error("invalid upstream credentials: {0}")]
    InvalidUpstreamCredentials(String),
    #[error("transport error: {0}")]
    Transport(String),
    #[error("upstream cancelled the response mid-stream")]
    StreamAborted,
}

impl BridgeError {
    /// Convenience constructor for synthesised upstream errors (tests,
    /// cooldown fixtures) where no real upstream envelope is involved.
    /// Sets [`UpstreamWire::Unknown`] and `parsed: None`.
    pub fn upstream_status(status: u16, message: impl Into<String>) -> Self {
        Self::UpstreamStatus {
            status,
            message: message.into(),
            parsed: None,
            wire: UpstreamWire::Unknown,
            retry_after: None,
            diagnostic: None,
        }
    }

    /// Convenience constructor for synthesised upstream errors that
    /// carry a parsed `Retry-After` hint. See [`upstream_status`].
    pub fn upstream_status_with_retry_after(
        status: u16,
        message: impl Into<String>,
        retry_after: Option<Duration>,
    ) -> Self {
        Self::UpstreamStatus {
            status,
            message: message.into(),
            parsed: None,
            wire: UpstreamWire::Unknown,
            retry_after,
            diagnostic: None,
        }
    }

    /// Have `Display` name the read failure behind this `UpstreamStatus`. What
    /// the caller is sent stays whatever `message` says: the caller's answer
    /// does not depend on how far the body got.
    pub fn with_body_read_error(mut self, reason: Option<String>) -> Self {
        if let (
            Self::UpstreamStatus {
                status, diagnostic, ..
            },
            Some(reason),
        ) = (&mut self, reason)
        {
            *diagnostic = Some(format!(
                "upstream returned HTTP {status}: failed to read response body: {reason}"
            ));
        }
        self
    }

    /// Have `Display` report `detail` as this `UpstreamStatus`'s message, for
    /// a route whose caller is sent other text than what the upstream said.
    pub fn with_telemetry_detail(mut self, detail: &str) -> Self {
        if let Self::UpstreamStatus {
            status, diagnostic, ..
        } = &mut self
        {
            *diagnostic = Some(upstream_status_display(*status, detail, None));
        }
        self
    }

    /// The [`UpstreamBodyUnread`](Self::UpstreamBodyUnread) of a failed read.
    pub fn upstream_body_unread(status: u16, err: &reqwest::Error) -> Self {
        let reason = crate::upstream_http::transport_error_message(err);
        crate::response_tap::warn_error_body_unread(status, &reason);
        Self::UpstreamBodyUnread {
            status,
            decode_error: err.to_string(),
            reason,
        }
    }

    /// The text of an error frame ending a stream. Written out rather than
    /// taken from `Display`, which is free to say more than the caller is told.
    pub fn stream_frame_message(&self) -> String {
        match self {
            Self::UpstreamStatus {
                status, message, ..
            } => format!("upstream returned HTTP {status}: {message}"),
            Self::UpstreamInBand { message, .. } => {
                format!("upstream reported an in-band stream error: {message}")
            }
            other => other.to_string(),
        }
    }
}

fn upstream_status_display(status: u16, message: &str, diagnostic: Option<&str>) -> String {
    match diagnostic {
        Some(diagnostic) => diagnostic.to_string(),
        None if message.chars().all(|c| c.is_control() || c.is_whitespace()) => {
            format!("upstream returned HTTP {status} with no error detail in the response")
        }
        None => format!("upstream returned HTTP {status}: {message}"),
    }
}

/// Says nothing of a stream: a whole 2xx body reports a failure this way too.
const IN_BAND_ERROR: &str = "upstream reported an in-band error";

fn in_band_display(message: &str, parsed: Option<&UpstreamErrorView>) -> String {
    if !message.chars().all(|c| c.is_control() || c.is_whitespace()) {
        return format!("{IN_BAND_ERROR}: {message}");
    }
    let named = parsed.and_then(|p| p.code.as_deref().or(p.kind.as_deref()));
    match named {
        Some(name) => format!("{IN_BAND_ERROR} with no message: {name}"),
        None => format!("{IN_BAND_ERROR} with no message"),
    }
}

/// Bound `read`, the read of a non-2xx body, by `deadline` on a path whose
/// request-level deadline ended with the response head, as a streaming one's
/// does: an upstream that stalls after its headers holds the caller for one more
/// budget at most. Past it the head is still known, so the answer is its status
/// and `Retry-After` with nothing read, the failure of a body cut off mid-read.
pub async fn upstream_error_within<F: std::future::Future<Output = BridgeError>>(
    deadline: Option<Duration>,
    resp: reqwest::Response,
    read: impl FnOnce(reqwest::Response) -> F,
) -> BridgeError {
    let Some(d) = deadline else {
        return read(resp).await;
    };
    let status = resp.status().as_u16();
    let retry_after = parse_retry_after(resp.headers());
    tokio::time::timeout(d, read(resp))
        .await
        .unwrap_or_else(|_| {
            let reason = format!("timed out after {}ms", d.as_millis());
            crate::response_tap::warn_error_body_unread(status, &reason);
            BridgeError::upstream_status_with_retry_after(status, "", retry_after)
                .with_body_read_error(Some(reason))
        })
}

/// Read a non-2xx answer a proxy route holds itself into the
/// `UpstreamStatus` a bridge would give it: the envelope `parse` recognises
/// for the client's `code` and `param`, and the upstream's own text as the
/// message whichever shape carried it. A body that could not be read whole
/// parses to nothing, so these routes answer such a failure with the status
/// fallback, not with a fragment.
pub async fn upstream_status_parsed_from_response(
    resp: reqwest::Response,
    wire: UpstreamWire,
    parse: impl FnOnce(&[u8]) -> Option<UpstreamErrorView>,
) -> BridgeError {
    let status = resp.status().as_u16();
    let retry_after = parse_retry_after(resp.headers());
    let body = read_body_capped(resp, MAX_UPSTREAM_ERROR_BODY_BYTES).await;
    upstream_status_of_body(status, body.whole(), retry_after, wire, parse)
        .with_body_read_error(body.read_error)
}

/// The `UpstreamStatus` of an error body: `parse`'s view of the envelope, and
/// its message or else what [`upstream_error_message`] reads out of the body.
pub fn upstream_status_of_body(
    status: u16,
    body: &[u8],
    retry_after: Option<Duration>,
    wire: UpstreamWire,
    parse: impl FnOnce(&[u8]) -> Option<UpstreamErrorView>,
) -> BridgeError {
    let parsed = parse(body).map(cap_error_view);
    let message = parsed
        .as_ref()
        .and_then(|v| v.message.clone())
        .filter(|message| !message.trim().is_empty())
        .unwrap_or_else(|| upstream_error_message(body));
    BridgeError::UpstreamStatus {
        status,
        message,
        parsed: parsed.map(Box::new),
        wire,
        retry_after,
        diagnostic: None,
    }
}

/// Parse the `Retry-After` response header into a Duration.
///
/// Per RFC 9110 §10.2.3, `Retry-After` may be either:
/// - a non-negative integer number of seconds, or
/// - an HTTP-date.
///
/// We accept the seconds form (which is what OpenAI / Anthropic /
/// DeepSeek / Gemini all return on 429). The HTTP-date form is rare
/// for AI providers and parsing it pulls in `httpdate`; skip for V1
/// — callers fall back to the configured default cooldown TTL.
///
/// Returns `None` when the header is absent, unparseable, or the
/// seconds value is unreasonable (the cooldown layer applies a
/// `max_seconds` clamp regardless).
pub fn parse_retry_after(headers: &http::HeaderMap) -> Option<Duration> {
    let raw = headers.get(http::header::RETRY_AFTER)?.to_str().ok()?;
    let seconds: u64 = raw.trim().parse().ok()?;
    Some(Duration::from_secs(seconds))
}

/// Drain an upstream error response (capped at
/// [`MAX_UPSTREAM_ERROR_BODY_BYTES`]) and produce a
/// [`BridgeError::UpstreamStatus`] with a best-effort parsed view of
/// the envelope.
///
/// The `parse` closure runs only when the response declares an
/// `application/json` content-type — this guards against fronting WAFs
/// or load balancers returning HTML error pages that would otherwise be
/// fed to a JSON parser and either fail expensively or surface
/// nonsensical fragments.
///
/// `parse` returning `None` is treated as "envelope shape unknown"; the
/// message then comes from [`upstream_error_message`], which reads the other
/// shapes an upstream's text arrives in and keeps a non-JSON body as sent.
pub async fn capture_upstream_error_http(
    status: http::StatusCode,
    resp: reqwest::Response,
    wire: UpstreamWire,
    parse: impl FnOnce(&[u8]) -> Option<UpstreamErrorView>,
) -> BridgeError {
    let retry_after = parse_retry_after(resp.headers());
    let UpstreamErrorBody {
        bytes: body,
        read_error,
    } = read_body_capped(resp, MAX_UPSTREAM_ERROR_BODY_BYTES).await;
    // Parse the error envelope opportunistically, regardless of the
    // upstream's Content-Type (#543). OpenAI's 401 `invalid_api_key`
    // path (and edge / proxy layers fronting some upstreams) return the
    // JSON error body labelled with a non-`application/json`
    // Content-Type; gating the parse on Content-Type silently dropped
    // `code` / `param` and dumped the raw body into `message`. The
    // per-bridge `parse` fn is the real validator — it requires the
    // provider's `{"error": {...}}` shape and returns `None` on any
    // non-matching body (HTML error pages, plain text, 5xx bodies), so
    // attempting it unconditionally is safe and strictly more robust.
    upstream_status_of_body(status.as_u16(), &body, retry_after, wire, parse)
        .with_body_read_error(read_error)
}

/// Truncate every parsed string at the same cap as the outer `message`.
/// Otherwise a hostile or buggy upstream emitting a 60 KB `error.message` /
/// `error.code` / `error.type` / `error.param` would reach the customer
/// envelope verbatim — the cap exists exactly to prevent that. AWS exception
/// codes / Anthropic types / OpenAI codes are bounded vocabulary in practice
/// but the cap applies defensively.
fn cap_error_view(mut v: UpstreamErrorView) -> UpstreamErrorView {
    let cap = MAX_UPSTREAM_ERROR_MESSAGE_BYTES;
    v.message = v.message.map(|m| truncate_lossy(&m, cap));
    v.kind = v.kind.map(|k| truncate_lossy(&k, cap));
    v.code = v.code.map(|c| truncate_lossy(&c, cap));
    v.param = v.param.map(|p| truncate_lossy(&p, cap));
    v
}

/// The human-readable message of an upstream error body: `error.message`
/// (the OpenAI / Google / Azure envelope), a top-level `message` (AWS), a
/// bare-string `error`, a `detail` string or `detail.message` (FastAPI-style
/// servers), or else the body text itself — truncated at
/// [`MAX_UPSTREAM_ERROR_MESSAGE_BYTES`]. The caller receives it as the
/// upstream wrote it.
pub fn upstream_error_message(body: &[u8]) -> String {
    upstream_error_text(body).unwrap_or_else(|| {
        truncate_lossy(
            &String::from_utf8_lossy(body),
            MAX_UPSTREAM_ERROR_MESSAGE_BYTES,
        )
    })
}

/// The text [`upstream_error_message`] reads out of a JSON error body, when the
/// body carries it in one of the shapes that function names.
pub fn upstream_error_text(body: &[u8]) -> Option<String> {
    let parsed: serde_json::Value = serde_json::from_slice(body).ok()?;
    let fields = [
        parsed.get("error").and_then(|e| e.get("message")),
        parsed.get("message"),
        parsed.get("error"),
        parsed.get("detail"),
        parsed.get("detail").and_then(|d| d.get("message")),
    ];
    let texts = || fields.iter().filter_map(|v| v.and_then(|m| m.as_str()));
    // A blank field does not hide the text a later one carries.
    texts()
        .find(|text| !text.trim().is_empty())
        .or_else(|| texts().next())
        .map(|message| truncate_lossy(message, MAX_UPSTREAM_ERROR_MESSAGE_BYTES))
}

/// Whether a top-level `error` value reports a failure: a non-empty
/// string, or an object with a non-empty `message` or a non-null `code`.
/// `null`, `{}`, `""` and an object carrying neither field do not — some
/// OpenAI-compatible servers send an empty `error` beside a successful
/// answer. The one rule every in-band probe applies, whether or not the
/// payload also carries `choices`.
pub fn in_band_error_reported(error: &serde_json::Value) -> bool {
    match error {
        serde_json::Value::String(s) => !s.is_empty(),
        serde_json::Value::Object(obj) => {
            let message = match obj.get("message") {
                None | Some(serde_json::Value::Null) => false,
                Some(serde_json::Value::String(s)) => !s.is_empty(),
                Some(_) => true,
            };
            message || obj.get("code").is_some_and(|c| !c.is_null())
        }
        _ => false,
    }
}

/// Probe one SSE `data:` payload, or a whole 2xx body, for a provider
/// in-band error envelope (`{"error": {...}}` or `{"error": "..."}`) and
/// capture it as a [`BridgeError::UpstreamInBand`]. Returns `None` when the
/// payload carries no `error` that [`in_band_error_reported`] accepts —
/// callers fall back to their decode path, so a genuinely malformed frame
/// still surfaces as [`BridgeError::UpstreamDecode`].
///
/// Field handling is tolerant across the OpenAI-compatible family and
/// Google's error shape:
/// - `error.type` (OpenAI / Anthropic vocabulary) populates the view's
///   `kind`; absent that, Google's `error.status` gRPC token
///   (`UNAVAILABLE`, `RESOURCE_EXHAUSTED`, …) does, so the per-wire
///   translation tables in `error_translate` apply either way.
/// - `error.code` may be a JSON number (Google, OpenRouter) or a string
///   (OpenAI). A numeric value in 400..=599 — either form — becomes the
///   embedded `status`; a non-numeric string stays in the view's `code`.
/// - a bare-string `error` value becomes the message (OpenAI-compatible
///   aggregators send this form).
///
/// All captured strings are truncated at
/// [`MAX_UPSTREAM_ERROR_MESSAGE_BYTES`], same as
/// [`capture_upstream_error_http`].
pub fn capture_in_band_error(payload: &str, wire: UpstreamWire) -> Option<BridgeError> {
    let value: serde_json::Value = serde_json::from_str(payload).ok()?;
    capture_in_band_error_value(&value, wire)
}

/// [`capture_in_band_error`] over a whole 2xx response body. The parse is
/// skipped unless the body mentions an `"error"` key, so a success body
/// costs one substring scan.
pub fn in_band_error_in_body(body: &[u8], wire: UpstreamWire) -> Option<BridgeError> {
    capture_in_band_error_value(&in_band_error_body(body)?, wire)
}

/// The parsed 2xx body, when its top-level `error` reports a failure by
/// [`in_band_error_reported`] — for a caller that reads the envelope with
/// its own wire's taxonomy. The parse is skipped unless the body mentions
/// an `"error"` key.
pub fn in_band_error_body(body: &[u8]) -> Option<serde_json::Value> {
    const KEY: &[u8] = b"\"error\"";
    if !body.windows(KEY.len()).any(|w| w == KEY) {
        return None;
    }
    let value: serde_json::Value = serde_json::from_slice(body).ok()?;
    value
        .get("error")
        .is_some_and(in_band_error_reported)
        .then_some(value)
}

/// Read a 2xx body and decode it as the success shape `T`, unless it
/// reports an in-band error ([`in_band_error_in_body`]), which is returned
/// instead. A body that is neither is [`BridgeError::UpstreamDecode`].
pub async fn decode_success_body<T: serde::de::DeserializeOwned>(
    resp: reqwest::Response,
    wire: UpstreamWire,
) -> Result<T, BridgeError> {
    let body = resp
        .bytes()
        .await
        .map_err(|e| crate::upstream_http::upstream_body_read_error(&e))?;
    if let Some(err) = in_band_error_in_body(&body, wire) {
        return Err(err);
    }
    serde_json::from_slice(&body).map_err(|e| BridgeError::UpstreamDecode(e.to_string()))
}

/// [`capture_in_band_error`] over an already-parsed payload.
pub fn capture_in_band_error_value(
    value: &serde_json::Value,
    wire: UpstreamWire,
) -> Option<BridgeError> {
    let error = value.get("error")?;
    if !in_band_error_reported(error) {
        return None;
    }
    let cap = MAX_UPSTREAM_ERROR_MESSAGE_BYTES;
    let text = |v: Option<&serde_json::Value>| match v {
        Some(serde_json::Value::String(s)) => Some(truncate_lossy(s, cap)),
        _ => None,
    };
    let (status, message, parsed) = match error {
        serde_json::Value::String(s) => (None, truncate_lossy(s, cap), None),
        inner => {
            let code = inner.get("code");
            let numeric_status = match code {
                Some(serde_json::Value::Number(n)) => n.as_u64(),
                Some(serde_json::Value::String(s)) => s.parse::<u64>().ok(),
                _ => None,
            }
            .and_then(|n| u16::try_from(n).ok())
            .filter(|n| (400..=599).contains(n));
            let message = match inner.get("message") {
                Some(serde_json::Value::String(s)) => Some(truncate_lossy(s, cap)),
                Some(serde_json::Value::Null) | None => None,
                Some(other) => Some(truncate_lossy(&other.to_string(), cap)),
            };
            let view = UpstreamErrorView {
                kind: text(inner.get("type")).or_else(|| text(inner.get("status"))),
                message,
                code: text(code).filter(|_| numeric_status.is_none()),
                param: text(inner.get("param")),
            };
            let message = view
                .message
                .clone()
                .unwrap_or_else(|| truncate_lossy(&error.to_string(), cap));
            (numeric_status, message, Some(Box::new(view)))
        }
    };
    Some(BridgeError::UpstreamInBand {
        status,
        message,
        parsed,
        wire,
    })
}

/// An upstream error response's body as far as it could be read.
pub struct UpstreamErrorBody {
    pub bytes: bytes::Bytes,
    /// Why reading stopped early, `None` when the body arrived whole.
    pub read_error: Option<String>,
}

impl UpstreamErrorBody {
    /// The bytes if the body arrived whole, nothing otherwise.
    pub fn whole(&self) -> &[u8] {
        if self.read_error.is_some() {
            &[]
        } else {
            &self.bytes
        }
    }
}

/// Read an upstream error response's body, keeping at most `limit` bytes.
/// Used to bound upstream-error parsing cost regardless of `Content-Length`.
/// THE read of every non-2xx upstream body: it hands the bytes to the request's
/// full-content exports ([`crate::response_tap::note_upstream_error_body`])
/// and logs a failed read.
///
/// Public so non-OpenAI / non-Anthropic bridges (Vertex, Azure) can
/// enforce the same cap when they need a custom parse path (e.g.
/// extracting only `kind` from the upstream envelope while suppressing
/// the `message` for operator-taxonomy redaction).
pub async fn read_body_capped(resp: reqwest::Response, limit: usize) -> UpstreamErrorBody {
    use futures::StreamExt;
    let status = resp.status().as_u16();
    let mut read_error = None;
    let mut buf = bytes::BytesMut::with_capacity(limit.min(16 * 1024));
    let mut stream = resp.bytes_stream();
    // Continue draining the stream past `limit` so the underlying
    // hyper connection can be returned to reqwest's keep-alive pool.
    // Stopping the iteration mid-stream taints the connection and
    // forces a new TCP handshake on the next upstream call — a real
    // cost when an upstream is flapping and producing a burst of
    // error responses. The extra reads only discard bytes; memory
    // stays bounded by `limit`.
    while let Some(chunk) = stream.next().await {
        let chunk = match chunk {
            Ok(chunk) => chunk,
            Err(e) => {
                let reason = crate::upstream_http::transport_error_message(&e);
                crate::response_tap::warn_error_body_unread(status, &reason);
                read_error = Some(reason);
                break;
            }
        };
        if buf.len() >= limit {
            continue;
        }
        let remaining = limit - buf.len();
        let take = chunk.len().min(remaining);
        buf.extend_from_slice(&chunk[..take]);
    }
    let bytes = buf.freeze();
    crate::response_tap::note_upstream_error_body(&bytes);
    UpstreamErrorBody { bytes, read_error }
}

/// Content-Type token starts with `application/json` (RFC 7231 §3.1.1.1
/// allows a trailing `; charset=…` parameter, so a prefix match is the
/// right shape here — exact equality misses `application/json; charset=utf-8`).
///
/// Public so non-OpenAI / non-Anthropic bridges (Vertex, Azure) can
/// apply the same JSON-only guard when they need a custom parse path
/// that doesn't route through [`capture_upstream_error_http`].
pub fn content_type_is_json(ct: &str) -> bool {
    let ct = ct.trim_start();
    ct.starts_with("application/json")
}

/// Convenience: read the `Content-Type` header from a [`reqwest::Response`]
/// and decide whether it's `application/json` per [`content_type_is_json`].
/// Returns `false` when the header is missing or non-ASCII.
pub fn response_is_json(resp: &reqwest::Response) -> bool {
    resp.headers()
        .get(http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|ct| content_type_is_json(&ct.to_ascii_lowercase()))
        .unwrap_or(false)
}

/// Truncate a string to at most `max` bytes, splitting only on a UTF-8
/// boundary. Appends an ellipsis when truncation occurred so log
/// readers can tell the message was cut. Public so bridges building
/// [`BridgeError::UpstreamInBand`] from typed (non-JSON-probe) sources
/// apply the same cap as [`capture_upstream_error_http`].
pub fn truncate_lossy(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &s[..end])
}

impl BridgeError {
    /// Stable HTTP status mapping. The proxy layer uses this to build
    /// its OpenAI-compatible `{error:{message,type,...}}` envelope.
    pub fn http_status(&self) -> u16 {
        match self {
            BridgeError::Timeout { .. } => 504,
            BridgeError::UpstreamStatus { status, .. } => {
                // We only forward 4xx directly; everything else collapses
                // to 502 so clients don't see upstream 5xx bleed through.
                if (400..500).contains(status) {
                    *status
                } else {
                    502
                }
            }
            BridgeError::UpstreamDecode(_)
            | BridgeError::UpstreamBodyUnread { .. }
            | BridgeError::TokenExchangeDecode(_) => 502,
            // Same forwarding rule as UpstreamStatus: an embedded 4xx is
            // the provider's judgment of the request and passes through;
            // 5xx / unknown collapse to 502.
            BridgeError::UpstreamInBand { status, .. } => match status {
                Some(s) if (400..500).contains(s) => *s,
                _ => 502,
            },
            // Same 500 as `Config`, which this variant was spelled as
            // before it was typed. The three routes that turn it into a
            // 501 build that response inline (they answer it as a dispatch
            // success, not through `ProxyError`), and the other two callers
            // of the `embed` default — the semantic router and the guardrail
            // embedder — surface it through this table. Mapping it to 501
            // here would change what those two return.
            BridgeError::Config(_) | BridgeError::UnsupportedCapability(_) => 500,
            BridgeError::InvalidUpstreamConfig(_) | BridgeError::InvalidRequest(_) => 400,
            BridgeError::RequestTooLarge { .. } => 413,
            BridgeError::InvalidUpstreamCredentials(_) => 401,
            BridgeError::Transport(_) => 502,
            BridgeError::StreamAborted => 502,
        }
    }

    /// Whether the request actually left for the upstream before this
    /// error was raised.
    ///
    /// Gates the `aisix_deployment_*` families, which read as **upstream
    /// health** for one deployment target. The config/credential/request
    /// variants are raised while the request is still being assembled — an
    /// empty or unusable `api_key`, a missing `model_name`/`api_base`, a
    /// body that would not serialize, a `split_system` shape the provider
    /// cannot express — so no provider was ever contacted, and counting
    /// them against a deployment reports our own misconfiguration as
    /// provider degradation.
    ///
    /// `Timeout` and `Transport` stay `true` on purpose: a connect timeout
    /// or a refused connection means we did try to reach the upstream, and
    /// "unreachable" is exactly the kind of health this family exists to
    /// show. Kept exhaustive (like [`http_status`](Self::http_status) and
    /// `routing_error_class`) so a new variant has to declare which side of
    /// the network boundary it sits on instead of inheriting a default.
    pub fn reached_upstream(&self) -> bool {
        match self {
            BridgeError::Timeout { .. }
            | BridgeError::UpstreamStatus { .. }
            | BridgeError::UpstreamDecode(_)
            | BridgeError::UpstreamBodyUnread { .. }
            | BridgeError::TokenExchangeDecode(_)
            | BridgeError::UpstreamInBand { .. }
            | BridgeError::Transport(_)
            | BridgeError::StreamAborted => true,
            BridgeError::Config(_)
            | BridgeError::UnsupportedCapability(_)
            | BridgeError::InvalidUpstreamConfig(_)
            | BridgeError::InvalidRequest(_)
            | BridgeError::RequestTooLarge { .. }
            | BridgeError::InvalidUpstreamCredentials(_) => false,
        }
    }

    /// Stable error-type token for the error envelope's `type` field.
    pub fn error_type(&self) -> &'static str {
        match self {
            BridgeError::Timeout { .. } => "timeout",
            BridgeError::UpstreamStatus { .. } => "upstream_error",
            BridgeError::UpstreamDecode(_)
            | BridgeError::UpstreamBodyUnread { .. }
            | BridgeError::TokenExchangeDecode(_) => "upstream_decode_error",
            BridgeError::UpstreamInBand { .. } => "upstream_in_band_error",
            BridgeError::Config(_) | BridgeError::UnsupportedCapability(_) => "config_error",
            BridgeError::InvalidUpstreamConfig(_)
            | BridgeError::InvalidRequest(_)
            | BridgeError::RequestTooLarge { .. } => "invalid_request_error",
            BridgeError::InvalidUpstreamCredentials(_) => "authentication_error",
            BridgeError::Transport(_) => "transport_error",
            BridgeError::StreamAborted => "stream_aborted",
        }
    }
}

/// A live stream of chunks. Boxed so the Bridge trait stays object-safe
/// (the Hub holds `Arc<dyn Bridge>` values).
pub type ChatChunkStream = BoxStream<'static, Result<ChatChunk, BridgeError>>;

/// What a platform answered an Anthropic Messages request with, in the
/// Anthropic Messages wire itself.
pub enum NativeMessages {
    /// The Messages response object.
    Json(serde_json::Value),
    /// Anthropic SSE frames (`event:` + `data:` + blank line), each item
    /// one or more whole frames. A failure inside the committed stream
    /// arrives as an Anthropic `error` frame; an `Err` item is a transport
    /// failure the bridge could not express as one.
    Stream(BoxStream<'static, Result<bytes::Bytes, BridgeError>>),
}

/// The provider-agnostic chat operation. Implementors live in the
/// individual `aisix-provider-*` crates.
#[async_trait]
pub trait Bridge: Send + Sync + 'static {
    /// Human-readable name used in logs and metrics labels. Stable across
    /// upgrades so dashboards don't break.
    fn name(&self) -> &'static str;

    /// The upstream API protocol this bridge speaks on the wire, as the
    /// bounded `upstream_protocol` metric label value
    /// (AISIX-Cloud#1403).
    ///
    /// NOT [`Bridge::name`]: a bridge names itself after the vendor
    /// family it serves, which for Bedrock varies per model family
    /// (`"anthropic"`, `"meta"`, …) while the wire shape stays
    /// `"bedrock"` throughout.
    ///
    /// The default is [`crate::hub::UPSTREAM_PROTOCOL_UNKNOWN`] so a
    /// stub bridge in a test does not have to answer. Every bridge
    /// reachable from `build_hub()` overrides it, which
    /// `upstream_protocol_label_matches_dispatched_bridge` pins.
    fn wire_protocol(&self) -> &'static str {
        crate::hub::UPSTREAM_PROTOCOL_UNKNOWN
    }

    /// Whether the upstream `ctx` dispatches to reads `cache_control`
    /// prompt-cache markers carried in the [`ChatFormat`] — on content
    /// parts, on a `role:"tool"` message, on `tool_calls[]` entries and on
    /// tool definitions. A translation from a protocol whose markers the
    /// chat form does not carry by default (Anthropic Messages) keeps them
    /// only when this is true; everywhere else they would reach an upstream
    /// that rejects or ignores them. The default is `false`.
    fn reads_cache_markers(&self, _ctx: &BridgeContext) -> bool {
        false
    }

    /// Non-streaming call: one request, one response.
    async fn chat(
        &self,
        req: &ChatFormat,
        ctx: &BridgeContext,
    ) -> Result<ChatResponse, BridgeError>;

    /// Streaming call: one request, a stream of deltas.
    async fn chat_stream(
        &self,
        req: &ChatFormat,
        ctx: &BridgeContext,
    ) -> Result<ChatChunkStream, BridgeError>;

    /// [`Bridge::chat`] for a caller that can hand over a request it will
    /// not reuse — a per-attempt translation or rewrite. An owned request
    /// is dropped as soon as the upstream body is built, so it is not
    /// alive for the upstream wait next to the body. Every production
    /// bridge overrides this; the default only borrows.
    async fn chat_cow(
        &self,
        req: std::borrow::Cow<'_, ChatFormat>,
        ctx: &BridgeContext,
    ) -> Result<ChatResponse, BridgeError> {
        self.chat(&req, ctx).await
    }

    /// [`Bridge::chat_stream`] with the same ownership contract as
    /// [`Bridge::chat_cow`].
    async fn chat_stream_cow(
        &self,
        req: std::borrow::Cow<'_, ChatFormat>,
        ctx: &BridgeContext,
    ) -> Result<ChatChunkStream, BridgeError> {
        self.chat_stream(&req, ctx).await
    }

    /// Embedding call: text(s) → float vectors. Providers that do not
    /// support embeddings keep the default, which returns
    /// [`BridgeError::UnsupportedCapability`] so `/v1/embeddings` can
    /// surface a 501 rather than a 502.
    async fn embed(
        &self,
        _req: &EmbeddingRequest,
        _ctx: &BridgeContext,
    ) -> Result<EmbeddingResponse, BridgeError> {
        Err(BridgeError::UnsupportedCapability(
            BridgeCapability::Embeddings,
        ))
    }

    /// Legacy text completions passthrough (`/v1/completions`).
    ///
    /// The request body JSON is forwarded verbatim after replacing the
    /// `model` field with the upstream provider model id. The response
    /// body JSON is returned as-is from the upstream so format differences
    /// between providers are the caller's responsibility.
    ///
    /// Providers that do not expose a `/completions` endpoint should keep
    /// the default, which returns
    /// [`BridgeError::UnsupportedCapability`] — the value `/v1/completions`
    /// turns into a 501.
    async fn complete(
        &self,
        _body: &serde_json::Value,
        _ctx: &BridgeContext,
    ) -> Result<serde_json::Value, BridgeError> {
        Err(BridgeError::UnsupportedCapability(
            BridgeCapability::TextCompletions,
        ))
    }

    /// Image generation passthrough (`/v1/images/generations`).
    ///
    /// The request body JSON is forwarded verbatim after replacing the
    /// `model` field with the upstream provider model id. The response
    /// body JSON is returned as-is from the upstream.
    ///
    /// Providers that do not expose an image generation endpoint should keep
    /// the default, which returns
    /// [`BridgeError::UnsupportedCapability`] — the value
    /// `/v1/images/generations` turns into a 501.
    async fn generate_image(
        &self,
        _body: &serde_json::Value,
        _ctx: &BridgeContext,
    ) -> Result<serde_json::Value, BridgeError> {
        Err(BridgeError::UnsupportedCapability(
            BridgeCapability::ImageGeneration,
        ))
    }

    /// Whether `upstream_model` is a Claude model this platform serves on
    /// its own Anthropic Messages routes, so `/v1/messages` reaches it
    /// through [`Bridge::anthropic_messages`] rather than the chat
    /// translation.
    fn serves_anthropic_messages(&self, _upstream_model: &str) -> bool {
        false
    }

    /// An Anthropic Messages body sent to the platform's own Claude route
    /// and answered in the same wire. `body` is the caller's request with
    /// the gateway's own rewrites applied; the bridge adapts it to what the
    /// platform accepts. `body.stream` picks the streaming route.
    async fn anthropic_messages(
        &self,
        _body: serde_json::Value,
        _ctx: &BridgeContext,
    ) -> Result<NativeMessages, BridgeError> {
        Err(BridgeError::UnsupportedCapability(
            BridgeCapability::AnthropicMessages,
        ))
    }

    /// `/v1/messages/count_tokens` on the platform's own token-counting
    /// API, answered as Anthropic's `{"input_tokens": N}`.
    async fn anthropic_count_tokens(
        &self,
        _body: serde_json::Value,
        _ctx: &BridgeContext,
    ) -> Result<serde_json::Value, BridgeError> {
        Err(BridgeError::UnsupportedCapability(
            BridgeCapability::AnthropicMessages,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An in-band error with an empty message says so where telemetry reads
    /// it; the stream's error frame keeps what the caller has always been sent.
    #[test]
    fn a_blank_in_band_message_is_named_without_changing_the_callers_frame() {
        let in_band = |payload: serde_json::Value| {
            capture_in_band_error_value(&payload, UpstreamWire::OpenAI).unwrap()
        };
        let with_code =
            in_band(serde_json::json!({"error": {"message": "", "code": "overloaded"}}));
        assert_eq!(
            with_code.to_string(),
            "upstream reported an in-band error with no message: overloaded"
        );
        assert_eq!(
            with_code.stream_frame_message(),
            "upstream reported an in-band stream error: "
        );
        let bare = BridgeError::UpstreamInBand {
            status: None,
            message: " \n".into(),
            parsed: None,
            wire: UpstreamWire::OpenAI,
        };
        assert_eq!(
            bare.to_string(),
            "upstream reported an in-band error with no message"
        );
        let named = in_band(serde_json::json!({"error": {"message": "boom"}}));
        assert_eq!(
            named.to_string(),
            "upstream reported an in-band error: boom"
        );
        assert_eq!(
            named.stream_frame_message(),
            "upstream reported an in-band stream error: boom"
        );
    }

    #[test]
    fn a_blank_field_does_not_hide_the_text_of_a_later_one() {
        assert_eq!(
            upstream_error_message(br#"{"message":"","error":"quota exceeded"}"#),
            "quota exceeded"
        );
        assert_eq!(upstream_error_message(br#"{"message":""}"#), "");
    }

    /// The wording of a blank message is the error's `Display`, and its
    /// `message` stays blank for the client's status fallback.
    #[test]
    fn upstream_status_display_names_what_a_blank_message_means() {
        for blank in ["", " \n", "\u{1}\u{2028}"] {
            let err = BridgeError::upstream_status(404, blank);
            assert_eq!(
                err.to_string(),
                "upstream returned HTTP 404 with no error detail in the response"
            );
        }
        let err = BridgeError::upstream_status(502, "fragment")
            .with_body_read_error(Some("connection reset".into()));
        assert_eq!(
            err.to_string(),
            "upstream returned HTTP 502: failed to read response body: connection reset"
        );
        // The caller's text is untouched by the diagnosis.
        let BridgeError::UpstreamStatus { message, .. } = err else {
            unreachable!()
        };
        assert_eq!(message, "fragment");
        assert_eq!(
            BridgeError::upstream_status(502, "upstream error")
                .with_telemetry_detail("real detail")
                .to_string(),
            "upstream returned HTTP 502: real detail"
        );
        assert_eq!(
            BridgeError::upstream_status(502, "upstream error")
                .with_telemetry_detail("")
                .to_string(),
            "upstream returned HTTP 502 with no error detail in the response"
        );
        // An error frame keeps the plain wording whatever `Display` says.
        assert_eq!(
            BridgeError::upstream_status(500, "").stream_frame_message(),
            "upstream returned HTTP 500: "
        );
        assert_eq!(
            BridgeError::upstream_status(400, "bad param").to_string(),
            "upstream returned HTTP 400: bad param"
        );
    }
    use crate::chat::{
        ChatFormat, ChatMessage, ChatResponse, EmbeddingRequest, FinishReason, UsageStats,
    };

    /// A Bridge that implements only what the trait requires, so the three
    /// optional operations fall through to their defaults — the shape every
    /// provider that does not speak embeddings / text completions / image
    /// generation actually has.
    struct ChatOnlyBridge;

    #[async_trait::async_trait]
    impl Bridge for ChatOnlyBridge {
        fn name(&self) -> &'static str {
            "chat-only"
        }
        async fn chat(
            &self,
            _req: &ChatFormat,
            _ctx: &BridgeContext,
        ) -> Result<ChatResponse, BridgeError> {
            Ok(ChatResponse {
                logprobs: None,
                more_choices: Vec::new(),
                id: String::new(),
                model: String::new(),
                message: ChatMessage::assistant(String::new()),
                finish_reason: FinishReason::Stop,
                usage: UsageStats::default(),
                service_tier: None,
            })
        }
        async fn chat_stream(
            &self,
            _req: &ChatFormat,
            _ctx: &BridgeContext,
        ) -> Result<ChatChunkStream, BridgeError> {
            unreachable!("not exercised")
        }
    }

    /// The three defaults name their capability in a typed field rather
    /// than only in prose. Before #1093 the proxy read the sentence to
    /// decide the status, so rewording it would have moved a 501 to a 500
    /// with nothing to catch it — and this test would not have existed to
    /// notice, because the message was the contract.
    ///
    /// The `Display` text is asserted too, because the three routes render
    /// it into the client-facing 501 envelope.
    #[tokio::test]
    async fn the_optional_operations_default_to_a_typed_capability_gap() {
        let b = ChatOnlyBridge;
        let ctx = BridgeContext::new(
            "req-1",
            std::sync::Arc::new(
                serde_json::from_str::<Model>(
                    r#"{"display_name":"m","provider":"anthropic","model_name":"claude","provider_key_id":"pk"}"#,
                )
                .unwrap(),
            ),
            std::sync::Arc::new(
                serde_json::from_str::<ProviderKey>(r#"{"display_name":"pk","secret":"s"}"#).unwrap(),
            ),
        );
        let embed_req = EmbeddingRequest {
            extra: Default::default(),
            token_input: None,
            model: "m".into(),
            input: vec!["hello".into()],
            input_was_single: true,
            encoding_format: None,
            dimensions: None,
            input_type: None,
            embedding_types: None,
        };

        let cases: Vec<(BridgeError, BridgeCapability, &str)> = vec![
            (
                b.embed(&embed_req, &ctx).await.unwrap_err(),
                BridgeCapability::Embeddings,
                "this provider does not support embeddings",
            ),
            (
                b.complete(&serde_json::json!({}), &ctx).await.unwrap_err(),
                BridgeCapability::TextCompletions,
                "this provider does not support text completions",
            ),
            (
                b.generate_image(&serde_json::json!({}), &ctx)
                    .await
                    .unwrap_err(),
                BridgeCapability::ImageGeneration,
                "this provider does not support image generation",
            ),
        ];

        for (err, capability, message) in cases {
            assert!(
                matches!(err, BridgeError::UnsupportedCapability(c) if c == capability),
                "expected {capability:?}, got {err}"
            );
            assert_eq!(err.to_string(), message);
            // The response shape the two non-route callers see is unchanged:
            // this stays exactly what `Config` reported.
            assert_eq!(err.http_status(), 500);
            assert_eq!(err.error_type(), "config_error");
            assert!(!err.reached_upstream());
        }
    }

    #[test]
    fn in_band_probe_parses_openai_string_code_envelope() {
        let e = capture_in_band_error(
            r#"{"error":{"message":"The server is overloaded","type":"server_error","code":"overloaded"}}"#,
            UpstreamWire::OpenAI,
        )
        .expect("error envelope must be captured");
        match e {
            BridgeError::UpstreamInBand {
                status,
                message,
                parsed,
                wire,
            } => {
                assert_eq!(status, None, "non-numeric code carries no status");
                assert_eq!(message, "The server is overloaded");
                let view = parsed.expect("view");
                assert_eq!(view.kind.as_deref(), Some("server_error"));
                assert_eq!(view.code.as_deref(), Some("overloaded"));
                assert!(matches!(wire, UpstreamWire::OpenAI));
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn in_band_probe_parses_google_numeric_code_and_grpc_status() {
        let e = capture_in_band_error(
            r#"{"error":{"code":503,"message":"The service is currently unavailable.","status":"UNAVAILABLE"}}"#,
            UpstreamWire::Vertex,
        )
        .expect("google error shape must be captured");
        match e {
            BridgeError::UpstreamInBand { status, parsed, .. } => {
                assert_eq!(status, Some(503));
                // The gRPC token lands in `kind` so the Vertex
                // translation table applies.
                assert_eq!(parsed.expect("view").kind.as_deref(), Some("UNAVAILABLE"));
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn in_band_probe_parses_numeric_string_code_as_status() {
        let e = capture_in_band_error(
            r#"{"error":{"code":"429","message":"rate limited"}}"#,
            UpstreamWire::OpenAI,
        )
        .expect("captured");
        match e {
            BridgeError::UpstreamInBand { status, parsed, .. } => {
                assert_eq!(status, Some(429));
                // A numeric code became the status; it is not repeated
                // in the view's string `code`.
                assert_eq!(parsed.expect("view").code, None);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn in_band_probe_accepts_bare_string_error() {
        let e = capture_in_band_error(r#"{"error":"boom"}"#, UpstreamWire::OpenAI)
            .expect("bare-string error form must be captured");
        match e {
            BridgeError::UpstreamInBand {
                status,
                message,
                parsed,
                ..
            } => {
                assert_eq!(status, None);
                assert_eq!(message, "boom");
                assert!(parsed.is_none());
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn in_band_probe_rejects_non_error_payloads() {
        assert!(capture_in_band_error(r#"{"id":"chunk-1"}"#, UpstreamWire::OpenAI).is_none());
        assert!(capture_in_band_error("not json at all", UpstreamWire::OpenAI).is_none());
        // An out-of-HTTP-range numeric code is not a status…
        let e = capture_in_band_error(
            r#"{"error":{"code":200,"message":"odd"}}"#,
            UpstreamWire::OpenAI,
        )
        .expect("still an error envelope");
        assert!(matches!(
            e,
            BridgeError::UpstreamInBand { status: None, .. }
        ));
    }

    #[test]
    fn in_band_probe_ignores_an_error_that_reports_nothing() {
        for payload in [
            r#"{"error":null}"#,
            r#"{"error":{}}"#,
            r#"{"error":""}"#,
            r#"{"error":{"message":"","code":null,"type":"x"}}"#,
            r#"{"id":"c","choices":[{"index":0}],"error":{}}"#,
        ] {
            assert!(
                capture_in_band_error(payload, UpstreamWire::OpenAI).is_none(),
                "{payload}"
            );
            assert!(in_band_error_in_body(payload.as_bytes(), UpstreamWire::OpenAI).is_none());
        }
        // A code alone, or a message beside answers, still reports one.
        for payload in [
            r#"{"error":{"code":502}}"#,
            r#"{"id":"c","choices":[],"error":{"code":502,"message":"down"}}"#,
            r#"{"choices":[{"index":0}],"error":{"message":"down"}}"#,
        ] {
            assert!(
                in_band_error_in_body(payload.as_bytes(), UpstreamWire::OpenAI).is_some(),
                "{payload}"
            );
        }
    }

    #[test]
    fn in_band_probe_caps_oversized_fields() {
        let long = "x".repeat(4 * MAX_UPSTREAM_ERROR_MESSAGE_BYTES);
        let payload = format!(r#"{{"error":{{"message":"{long}","type":"{long}"}}}}"#);
        let e = capture_in_band_error(&payload, UpstreamWire::OpenAI).expect("captured");
        match e {
            BridgeError::UpstreamInBand {
                message, parsed, ..
            } => {
                assert!(message.len() <= MAX_UPSTREAM_ERROR_MESSAGE_BYTES + '…'.len_utf8());
                let view = parsed.expect("view");
                assert!(
                    view.kind.unwrap().len() <= MAX_UPSTREAM_ERROR_MESSAGE_BYTES + '…'.len_utf8()
                );
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn in_band_error_status_forwarding_matches_upstream_status_rule() {
        let mk = |status: Option<u16>| BridgeError::UpstreamInBand {
            status,
            message: "m".into(),
            parsed: None,
            wire: UpstreamWire::OpenAI,
        };
        assert_eq!(mk(Some(429)).http_status(), 429, "embedded 4xx forwards");
        assert_eq!(mk(Some(500)).http_status(), 502, "5xx collapses to 502");
        assert_eq!(mk(Some(529)).http_status(), 502);
        assert_eq!(mk(None).http_status(), 502);
        assert_eq!(mk(None).error_type(), "upstream_in_band_error");
    }

    #[test]
    fn timeout_maps_to_504() {
        let e = BridgeError::Timeout {
            cause: String::new(),
            elapsed_ms: 30_000,
        };
        assert_eq!(e.http_status(), 504);
        assert_eq!(e.error_type(), "timeout");
    }

    /// A gateway-owned deadline has no transport cause, and its message
    /// must stay byte-identical to what it was before `cause` existed —
    /// operators and log queries key on this sentence.
    #[test]
    fn timeout_without_cause_renders_unchanged() {
        let e = BridgeError::Timeout {
            elapsed_ms: 30_000,
            cause: String::new(),
        };
        assert_eq!(e.to_string(), "upstream request timed out after 30000ms");
    }

    /// With a transport cause the message names it, so an expired
    /// `connect_timeout` and an expired request budget stop looking
    /// identical (AISIX-Cloud#1093).
    #[test]
    fn timeout_with_cause_appends_it() {
        let e = BridgeError::Timeout {
            elapsed_ms: 5_001,
            cause: "error sending request: client error (Connect): tcp connect error: \
                    Connection timed out (os error 110)"
                .to_string(),
        };
        assert_eq!(
            e.to_string(),
            "upstream request timed out after 5001ms: error sending request: \
             client error (Connect): tcp connect error: Connection timed out (os error 110)"
        );
        // Status and telemetry class are unaffected by the added detail.
        assert_eq!(e.http_status(), 504);
        assert_eq!(e.error_type(), "timeout");
    }

    #[test]
    fn upstream_4xx_passes_through_5xx_collapses_to_502() {
        let e400 = BridgeError::upstream_status(429, "rate limit");
        assert_eq!(e400.http_status(), 429);

        let e500 = BridgeError::upstream_status(503, "busy");
        assert_eq!(e500.http_status(), 502);

        let e3xx = BridgeError::upstream_status(301, "redirect");
        // Non-4xx collapses too — redirects we don't follow are 502-worthy.
        assert_eq!(e3xx.http_status(), 502);
    }

    #[test]
    fn upstream_status_carries_retry_after_when_provided() {
        let e = BridgeError::upstream_status_with_retry_after(
            429,
            "slow down",
            Some(Duration::from_secs(60)),
        );
        match e {
            BridgeError::UpstreamStatus { retry_after, .. } => {
                assert_eq!(retry_after, Some(Duration::from_secs(60)));
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn parse_retry_after_handles_seconds_form() {
        let mut h = http::HeaderMap::new();
        h.insert(http::header::RETRY_AFTER, "30".parse().unwrap());
        assert_eq!(parse_retry_after(&h), Some(Duration::from_secs(30)));
    }

    #[test]
    fn parse_retry_after_returns_none_for_http_date_form() {
        let mut h = http::HeaderMap::new();
        h.insert(
            http::header::RETRY_AFTER,
            "Wed, 21 Oct 2026 07:28:00 GMT".parse().unwrap(),
        );
        // V1: HTTP-date form is intentionally not parsed.
        assert_eq!(parse_retry_after(&h), None);
    }

    #[test]
    fn parse_retry_after_returns_none_when_absent() {
        let h = http::HeaderMap::new();
        assert_eq!(parse_retry_after(&h), None);
    }

    #[test]
    fn transport_and_decode_errors_collapse_to_502() {
        assert_eq!(
            BridgeError::Transport("connection refused".into()).http_status(),
            502,
        );
        assert_eq!(
            BridgeError::UpstreamDecode("bad json".into()).http_status(),
            502,
        );
    }

    #[test]
    fn config_error_maps_to_500() {
        assert_eq!(
            BridgeError::Config("missing api_key".into()).http_status(),
            500
        );
        assert_eq!(
            BridgeError::Config("missing api_key".into()).error_type(),
            "config_error"
        );
    }

    #[test]
    fn invalid_upstream_config_maps_to_400_invalid_request() {
        // #367: customer-fixable config (missing api_base, missing
        // model_name, request-shape …) is a 400, not a 500 — retrying
        // won't help and a 5xx wrongly reads as a server fault.
        let e = BridgeError::InvalidUpstreamConfig("provider_key has no api_base".into());
        assert_eq!(e.http_status(), 400);
        assert_eq!(e.error_type(), "invalid_request_error");
    }

    #[test]
    fn invalid_upstream_credentials_maps_to_401_authentication() {
        // #367 follow-up: auth-material problems (empty/invalid secret,
        // unparseable credential JSON) are a 401 authentication_error,
        // not a 400 — they're a distinct class from request/routing shape.
        let e = BridgeError::InvalidUpstreamCredentials("provider_key.api_key is empty".into());
        assert_eq!(e.http_status(), 401);
        assert_eq!(e.error_type(), "authentication_error");
    }

    #[test]
    fn debugging_a_context_names_its_credentials_without_quoting_them() {
        let pk = sample_provider_key();
        let api_key = pk.api_key.clone();
        assert!(!api_key.is_empty(), "the sample must carry a real secret");

        let mut inbound = HeaderMap::new();
        inbound.insert(
            http::header::AUTHORIZATION,
            "Bearer header.payload.signature".parse().expect("header"),
        );

        let ctx = BridgeContext::new(
            "req-1",
            std::sync::Arc::new(sample_model()),
            std::sync::Arc::new(pk),
        )
        .with_client(
            CallerIdentity::default(),
            Some(std::sync::Arc::new(inbound)),
        );

        // The credential is really on the context: without this the
        // assertions below would pass on an empty one.
        assert_eq!(
            ctx.client_headers.as_ref().expect("headers")[http::header::AUTHORIZATION],
            "Bearer header.payload.signature"
        );

        let printed = format!("{ctx:?}");
        assert!(
            printed.contains("req-1") && printed.contains("authorization"),
            "the non-secret fields, header names included, still print: {printed}"
        );
        assert!(
            !printed.contains("header.payload.signature"),
            "the caller's token must not print: {printed}"
        );
        assert!(
            !printed.contains(&api_key),
            "the provider key's own secret must not print either: {printed}"
        );
    }

    #[test]
    fn context_defaults_no_deadline_with_helper_setter() {
        let m = std::sync::Arc::new(sample_model());
        let pk = std::sync::Arc::new(sample_provider_key());
        let ctx = BridgeContext::new("req-1", m.clone(), pk);
        assert_eq!(ctx.request_id, "req-1");
        assert!(ctx.deadline.is_none());
        let ctx = ctx.with_deadline(Duration::from_secs(30));
        assert_eq!(ctx.deadline, Some(Duration::from_secs(30)));
    }

    fn sample_model() -> Model {
        serde_json::from_str(
            r#"{
                "display_name": "test",
                "provider": "openai",
                "model_name": "gpt-4o",
                "provider_key_id": "11111111-1111-1111-1111-111111111111"
            }"#,
        )
        .unwrap()
    }

    fn sample_provider_key() -> ProviderKey {
        serde_json::from_str(r#"{"display_name":"openai-prod","secret":"sk-x"}"#).unwrap()
    }

    #[test]
    fn sample_model_resolves_to_openai() {
        let m = sample_model();
        assert_eq!(m.provider.as_deref(), Some("openai"));
    }
}
