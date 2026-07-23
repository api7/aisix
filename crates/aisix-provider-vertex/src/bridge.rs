//! `VertexBridge` — family Bridge for [`Adapter::Vertex`].
//!
//! Multi-publisher dispatch for Google Vertex AI. The publisher is
//! resolved from the upstream model id and routed to a per-publisher
//! wire path. **Currently wired:** `google` (Gemini) chat + streaming
//! (`:generateContent` / `:streamGenerateContent`); `anthropic`
//! (Claude) chat + streaming via `:rawPredict` / `:streamRawPredict`
//! (Anthropic Messages wire); the **OpenAI-compatible MaaS family**
//! (Llama, DeepSeek, Qwen, gpt-oss, MiniMax, Moonshot, Z.ai) chat +
//! streaming via the OpenAI shim (`endpoints/openapi/chat/completions`);
//! and **Mistral** (`mistralai`) + **AI21 Jamba** (`ai21`) chat +
//! streaming via the partner `:rawPredict` / `:streamRawPredict` URL with
//! an OpenAI-compatible body (model in BOTH the URL and the body). All
//! five Vertex publisher rails are now wired.
//!
//! Credentials: `ProviderKey.api_key` is a JSON-encoded
//! `{access_token, project, region}` struct. The `access_token` is
//! a pre-minted GCP OAuth2 bearer (operator-managed refresh; D5.1
//! follow-up adds in-process JWT-signing).
//!
//! URL pattern (Gemini, `generateContent`):
//! `https://<region>-aiplatform.googleapis.com/v1/projects/<project>/
//!  locations/<region>/publishers/google/models/<model>:generateContent`

use aisix_gateway::{
    sse::{SseDecoder, SseEvent},
    Bridge, BridgeContext, BridgeError, ChatChunk, ChatChunkStream, ChatDelta, ChatFormat,
    ChatMessage, ChatResponse, EmbeddingObject, EmbeddingRequest, EmbeddingResponse,
    EmbeddingUsage, EmbeddingVector, FinishReason, Role, UsageStats,
};
use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use http::{
    header::{HeaderName, HeaderValue},
    HeaderMap,
};
use reqwest::{header, Client, StatusCode};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::token_mint::{ServiceAccountKey, TokenMinter};
use crate::wire;

// Claude on Vertex reuses the shared Anthropic Messages serializer +
// response decoder (the same `wire` items the Bedrock `/invoke` path
// reuses). The Vertex `:rawPredict` body is Anthropic Messages JSON
// minus `model`/`stream` plus a Vertex-specific `anthropic_version`;
// `:streamRawPredict` keeps `stream: true` in the body and emits native
// Anthropic SSE, decoded by the shared `AnthropicStreamEvent` +
// `StreamState` (the same decoder the direct Anthropic bridge uses).
use aisix_provider_anthropic::wire::{
    build_request as build_anthropic_request, response_into_chat_response, split_system,
    AnthropicResponse, AnthropicStreamEvent, StreamState,
};

// Llama + the OpenAI-compatible MaaS family on Vertex use the OpenAI
// chat-completions shim, so reuse the OpenAI request serializer +
// response/stream decoders verbatim — the wire matches direct OpenAI.
use aisix_provider_openai::wire::{
    build_request as build_openai_request, messages_from as openai_messages_from,
    response_into_chat_response as openai_response_into_chat_response,
    stream_chunk_into_chat_chunk as openai_stream_chunk_into_chat_chunk, OpenAiResponse,
    OpenAiStreamChunk,
};

// Per-`ProviderKey` request/response override pipeline (#302 §5 / #339).
// The Vertex bridge mirrors `OpenAiBridge`'s apply order exactly and reuses
// the same primitives, so cp-api captures a provider's quirks once and every
// adapter honours the same wire shape. All primitives are no-ops when the
// targeted keys are absent, so they are safe to call uniformly across the
// five publisher rails (the Gemini `contents` shape simply does not match the
// OpenAI-style top-level keys the request transforms look for).
use aisix_core::RequestOverrides;
use aisix_provider_openai::overrides::{
    apply_content_list_to_string, apply_default_body_fields, apply_default_headers,
    apply_param_constraints, apply_param_renames,
};

/// `anthropic_version` value Vertex's Claude `:rawPredict` endpoint
/// requires in the request body. Distinct from the Bedrock value
/// (`bedrock-2023-05-31`). Per Google's Vertex AI Claude reference
/// <https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/use-claude>.
const VERTEX_ANTHROPIC_VERSION: &str = "vertex-2023-10-16";

/// Family Bridge for Google Vertex AI.
pub struct VertexBridge {
    client: Client,
    /// Static `name()` returned to the Hub. Stable across upgrades so
    /// metrics dashboards keep their existing `provider="vertex"`
    /// filters working.
    name: &'static str,
    /// In-process GCP OAuth2 token minter + cache. Used only on the
    /// `service_account_json` secret path; the pre-minted-token path
    /// bypasses this entirely. Wrapped in `Arc` so the bridge stays
    /// `Clone`-friendly for callers that share it across Hub registrations.
    token_minter: Arc<TokenMinter>,
    /// Test-only Vertex API base override (e.g. wiremock URI). When
    /// set, replaces the canonical `<region>-aiplatform.googleapis.com`
    /// host so wiremock can stand in.
    #[cfg(test)]
    api_base_override: Option<String>,
}

impl VertexBridge {
    /// Construct a Vertex bridge with the canonical name `"vertex"`.
    pub fn new() -> Self {
        Self::with_client(default_client())
    }

    /// Construct with a caller-supplied [`reqwest::Client`]. Useful
    /// when downstream callers want to share a connection pool.
    pub fn with_client(client: Client) -> Self {
        Self {
            token_minter: Arc::new(TokenMinter::new(client.clone())),
            client,
            name: "vertex",
            #[cfg(test)]
            api_base_override: None,
        }
    }

    /// Test-only seam: replace the canonical Vertex host with this
    /// URL (e.g. a wiremock URI). Credentials, project, region,
    /// SDK-equivalent URL stitching all run normally; only the
    /// destination host is different.
    #[cfg(test)]
    pub(crate) fn with_api_base_override(mut self, url: impl Into<String>) -> Self {
        self.api_base_override = Some(url.into());
        self
    }

    /// Test-only seam: replace the SA `token_uri` host on the
    /// internal token minter. Used by the SA-flow tests to redirect
    /// JWT-bearer assertions to a wiremock endpoint.
    #[cfg(test)]
    pub(crate) fn with_token_endpoint_override(mut self, url: impl Into<String>) -> Self {
        let new_minter = TokenMinter::new(self.client.clone()).with_token_endpoint_override(url);
        self.token_minter = Arc::new(new_minter);
        self
    }

    /// Resolve the base host the bridge POSTs to.
    ///
    /// Precedence (highest first):
    ///   1. `#[cfg(test)]` `Self::with_api_base_override` — test seam.
    ///   2. `ProviderKey.api_base` — production override for
    ///      corporate-proxy / private-VPC / mock deployments. The
    ///      operator-supplied value is trimmed and any trailing `/`
    ///      is stripped so `chat_gemini`'s URL stitching produces
    ///      a single-slash separator.
    ///   3. Canonical `https://<region>-aiplatform.googleapis.com`.
    ///
    /// Mirrors the Bedrock bridge's `endpoint_url` precedence
    /// (`crates/aisix-provider-bedrock/src/bridge.rs::build_client_from_ctx`).
    /// Fixes api7/ai-gateway#390 — pre-fix production builds
    /// silently dropped `ProviderKey.api_base` for Vertex, so a BYO
    /// operator's corporate-proxy URL was ignored.
    ///
    /// **Validation of `ProviderKey.api_base`** (PR #392 audit MEDIUM-1):
    /// the operator-supplied override is interpolated directly into
    /// the URL template, so we reject classes of input that the
    /// bridge cannot safely concatenate or that would shadow auth:
    ///   - non-`http(s)` schemes (no `file://`, `gs://`, etc.)
    ///   - userinfo `@` (an operator embedding `user:pass@host` would
    ///     leak credentials via logs / SSRF-style escalation)
    ///   - `?` query string (would silently merge with the streaming
    ///     `?alt=sse` the bridge appends)
    ///   - `#` fragment (no meaningful semantic for an upstream POST)
    ///
    /// Mirrors the Azure-OpenAI sibling fix (#391) which rejects the
    /// same classes. Errors surface as `BridgeError::Config` so the
    /// operator sees an actionable message instead of a silent
    /// fall-through-to-canonical (which is what the original #390 bug
    /// was — exactly what we don't want to bring back).
    fn resolve_api_base(
        &self,
        region: &str,
        ctx_api_base: Option<&str>,
    ) -> Result<String, BridgeError> {
        #[cfg(test)]
        if let Some(b) = &self.api_base_override {
            return Ok(b.clone());
        }
        if let Some(b) = ctx_api_base.map(str::trim).filter(|s| !s.is_empty()) {
            if !(b.starts_with("https://") || b.starts_with("http://")) {
                return Err(BridgeError::InvalidUpstreamConfig(format!(
                    "vertex provider_key api_base must use http:// or https:// scheme, got {b:?}",
                )));
            }
            if b.contains('@') {
                // Redact userinfo before echoing the rejected value into
                // the error string — the whole point of rejecting `@` is
                // that operator-pasted credentials shouldn't appear in
                // logs. Audit #392 re-audit LOW-1.
                let redacted = redact_userinfo(b);
                return Err(BridgeError::InvalidUpstreamConfig(format!(
                    "vertex provider_key api_base must not embed userinfo (@); use the request's \
                     Authorization header instead, got {redacted:?}",
                )));
            }
            if b.contains('?') {
                return Err(BridgeError::InvalidUpstreamConfig(format!(
                    "vertex provider_key api_base must not contain a query string (the bridge \
                     appends `?alt=sse` on streaming; an operator query would silently merge), \
                     got {b:?}",
                )));
            }
            if b.contains('#') {
                return Err(BridgeError::InvalidUpstreamConfig(format!(
                    "vertex provider_key api_base must not contain a fragment, got {b:?}",
                )));
            }
            // Reject an embedded path component — only `scheme://host[:port]`
            // is a valid origin. A bare trailing slash is fine (trimmed
            // below); a real path segment (e.g. `.../evil`) would silently
            // redirect every upstream call onto the wrong path, so fail fast
            // with a clear Config error. Backslashes are rejected too: the
            // WHATWG URL parser the HTTP client uses normalizes `\` to `/` on
            // http(s) URLs, so `host\evil` injects a path exactly like
            // `host/evil`. `b` has no `@`/`?`/`#` here (rejected above), so
            // echoing it is safe. Audit #434 LOW-1 / #435 (+ #464 audit MEDIUM).
            let after_scheme = b
                .split_once("://")
                .map(|(_, rest)| rest)
                .unwrap_or(b)
                .trim_end_matches('/');
            if after_scheme.contains('/') || after_scheme.contains('\\') {
                return Err(BridgeError::InvalidUpstreamConfig(format!(
                    "vertex provider_key api_base must be a bare origin \
                     (scheme://host[:port]) with no path, got {b:?}",
                )));
            }
            return Ok(b.trim_end_matches('/').to_string());
        }
        Ok(format!("https://{region}-aiplatform.googleapis.com"))
    }
}

impl Default for VertexBridge {
    fn default() -> Self {
        Self::new()
    }
}

fn default_client() -> Client {
    aisix_gateway::client_builder()
        .user_agent("aisix/0.1")
        .build()
        .unwrap_or_else(|_| Client::new())
}

/// The set of Vertex publishers we dispatch to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VertexPublisher {
    /// `publishers/google/models/gemini-*` — Google's own Gemini line.
    Google,
    /// `publishers/anthropic/models/claude-*` — Anthropic models hosted
    /// on Vertex. Wire shape is `rawPredict`, not canonical Anthropic
    /// Messages.
    Anthropic,
    /// The OpenAI-compatible MaaS family on Vertex — Meta Llama,
    /// DeepSeek, Qwen, gpt-oss, MiniMax, Moonshot, Z.ai. These do NOT
    /// use a `publishers/<vendor>/...:rawPredict` URL; Vertex exposes
    /// them through a single OpenAI chat-completions shim at
    /// `endpoints/openapi/chat/completions` (the model id goes in the
    /// request body, not the URL). One wire shape serves the whole
    /// family, so adding a new MaaS vendor is a resolver-prefix add.
    /// <https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/llama#openai>
    OpenAiCompat,
    /// `publishers/mistralai/models/mistral-*` — Mistral on Vertex
    /// (OpenAI-compatible body over the `:rawPredict` rail).
    Mistral,
    /// `publishers/ai21/models/jamba-*` — AI21 Jamba family
    /// (OpenAI-compatible body over the `:rawPredict` rail).
    Ai21,
}

impl VertexPublisher {
    /// Resolve the publisher from the upstream model id.
    pub fn from_upstream_id(upstream_id: &str) -> Option<Self> {
        let lower = upstream_id.to_ascii_lowercase();
        if lower.starts_with("gemini-") {
            Some(Self::Google)
        } else if lower.starts_with("claude-") {
            Some(Self::Anthropic)
        } else if lower.starts_with("meta/")
            || lower.starts_with("llama")
            || lower.starts_with("deepseek")
            || lower.starts_with("qwen")
            || lower.starts_with("openai/gpt-oss")
            || lower.starts_with("minimaxai/")
            || lower.starts_with("moonshotai/")
            || lower.starts_with("zai-org/")
        {
            // The OpenAI-shim MaaS family. Prefix set mirrors the
            // documented Vertex Model Garden MaaS models served via the
            // OpenAI chat-completions endpoint; Mistral / AI21 are
            // deliberately excluded (they use `rawPredict`, handled by
            // their own arms below).
            Some(Self::OpenAiCompat)
        } else if lower.starts_with("mistral-") || lower.starts_with("codestral-") {
            Some(Self::Mistral)
        } else if lower.starts_with("jamba-") {
            Some(Self::Ai21)
        } else {
            None
        }
    }

    /// The `publishers/<tag>` URL segment Vertex expects.
    ///
    /// **Returns `None` for [`Self::OpenAiCompat`]** — the MaaS family
    /// uses the OpenAI shim at `endpoints/openapi/chat/completions`,
    /// which carries no `publishers/<vendor>/...` segment.
    pub fn url_segment(self) -> Option<&'static str> {
        Some(match self {
            Self::Google => "publishers/google",
            Self::Anthropic => "publishers/anthropic",
            Self::Mistral => "publishers/mistralai",
            Self::Ai21 => "publishers/ai21",
            Self::OpenAiCompat => return None,
        })
    }
}

/// `ProviderKey.api_key` schema for a Vertex provider key.
///
/// Convention: GCP credentials are JSON-encoded into the `secret`
/// field. The operator chooses ONE of two credential modes:
///
/// 1. **Pre-minted token** — set `access_token`; operator manages
///    refresh (GCP token TTL ~1h). Backward-compatible with the
///    original D5.2.a schema; useful for short-lived testing
///    rigs and for operators who already have a token-mint pipeline.
///
/// 2. **In-process SA mint** (D5.1) — set `service_account_json`
///    to the full GCP service-account JSON key. The bridge signs a
///    JWT with the SA's RSA private key, exchanges it for an OAuth2
///    access token via the SA's `token_uri`, and caches it
///    in-process with TTL refresh.
///
/// Exactly one of the two must be set. Setting both — or neither —
/// fails at parse time so the operator gets an actionable error
/// before the first chat.
#[derive(Debug, Deserialize)]
struct VertexSecret {
    /// Pre-minted GCP OAuth2 access token (operator manages refresh).
    /// Mutually exclusive with `service_account_json`.
    #[serde(default)]
    access_token: Option<String>,
    /// GCP service-account JSON key (the on-disk shape `gcloud iam
    /// service-accounts keys create` emits). When present, the bridge
    /// mints + caches tokens in-process. Mutually exclusive with
    /// `access_token`.
    #[serde(default)]
    service_account_json: Option<ServiceAccountKey>,
    /// GCP project id (numeric or named, e.g. `my-org-prod`).
    project: String,
    /// GCP region the Vertex AI deployment targets
    /// (e.g. `us-central1`, `europe-west4`).
    region: String,
}

impl VertexSecret {
    /// Parse the JSON-encoded credential blob and validate the
    /// mutually-exclusive credential modes.
    ///
    /// **Audit-aware:** error messages MUST NOT echo raw secret
    /// bytes (serde error messages can leak partial content via
    /// "invalid character X at position N").
    fn parse(secret: &str) -> Result<Self, BridgeError> {
        if secret.trim().is_empty() {
            return Err(BridgeError::InvalidUpstreamCredentials(
                "vertex provider_key.api_key is empty — \
                 expected JSON with project, region, and either access_token \
                 or service_account_json"
                    .into(),
            ));
        }
        let parsed: VertexSecret = serde_json::from_str(secret).map_err(|_e| {
            BridgeError::InvalidUpstreamCredentials(
                "vertex provider_key.api_key must be valid JSON: \
                 {project, region, and either access_token or service_account_json}"
                    .into(),
            )
        })?;
        // Enforce mutual exclusion. Both-set is suspect (which one
        // wins?); neither-set is unusable. Empty-string token is a
        // distinct error so the operator gets a clearer message than
        // generic "neither set".
        if parsed.access_token.as_deref().is_some_and(str::is_empty) {
            return Err(BridgeError::InvalidUpstreamCredentials(
                "vertex provider_key.api_key.access_token is empty".into(),
            ));
        }
        let has_token = parsed
            .access_token
            .as_deref()
            .is_some_and(|t| !t.is_empty());
        let has_sa = parsed.service_account_json.is_some();
        if has_token && has_sa {
            return Err(BridgeError::InvalidUpstreamCredentials(
                "vertex provider_key.api_key must set exactly one of access_token \
                 or service_account_json (both were provided)"
                    .into(),
            ));
        }
        if !has_token && !has_sa {
            return Err(BridgeError::InvalidUpstreamCredentials(
                "vertex provider_key.api_key must set either access_token or \
                 service_account_json (neither was provided)"
                    .into(),
            ));
        }
        // Validate the SA shape eagerly if present so the operator
        // hits the actionable error at parse, not at first chat.
        if let Some(sa) = &parsed.service_account_json {
            sa.validate()?;
        }
        Ok(parsed)
    }

    /// Resolve the bearer token to use on this request. Returns the
    /// pre-minted token verbatim if set; otherwise mints (or pulls
    /// from cache) via the bridge's [`TokenMinter`].
    async fn resolve_access_token(&self, minter: &TokenMinter) -> Result<String, BridgeError> {
        if let Some(token) = &self.access_token {
            return Ok(token.clone());
        }
        if let Some(sa) = &self.service_account_json {
            return minter.get_token(sa).await;
        }
        // parse() rejects neither-set, so this is unreachable in
        // practice — keep the explicit error for defense in depth.
        Err(BridgeError::Config(
            "internal: VertexSecret has neither token nor SA after parse".into(),
        ))
    }
}

/// Validate that a path token (project id, region, model name) is
/// safe to interpolate into the Vertex URL. GCP project ids are
/// `[a-z][a-z0-9-]{4,28}[a-z0-9]`; region names are
/// `[a-z]+[0-9]+(-[a-z])?`; model names are vendor-pinned strings.
/// Reject `?`, `#`, `/`, whitespace, `..` so a malicious model_name
/// can't redirect dispatch.
fn validate_url_token(name: &str, value: &str) -> Result<(), BridgeError> {
    if value.is_empty() {
        return Err(BridgeError::Config(format!(
            "vertex {name} is empty (expected an identifier)"
        )));
    }
    if value.contains('/')
        || value.contains('?')
        || value.contains('#')
        || value.contains(' ')
        || value.contains('\t')
        || value.contains('\n')
        || value.contains("..")
    {
        return Err(BridgeError::Config(format!(
            "vertex {name} {value:?} contains URL-control characters — \
             reject `/`, `?`, `#`, whitespace, `..`"
        )));
    }
    Ok(())
}

/// Pull the upstream model id off the BridgeContext.
fn upstream_model(ctx: &BridgeContext) -> Result<&str, BridgeError> {
    ctx.model
        .model_name
        .as_deref()
        .ok_or_else(|| BridgeError::InvalidUpstreamConfig("model.model_name missing".into()))
}

/// Redact embedded userinfo from a URL string before echoing it
/// into an error message. Specifically: `scheme://user:pass@host`
/// becomes `scheme://<redacted>@host`. Only touches the substring
/// between `://` and the first `@`; leaves the rest of the URL
/// intact (including any other `@` in a path component). PR #392
/// re-audit LOW-1.
fn redact_userinfo(url: &str) -> String {
    let Some(scheme_end) = url.find("://") else {
        return url.to_string();
    };
    let rest_start = scheme_end + "://".len();
    let rest = &url[rest_start..];
    let Some(at_offset) = rest.find('@') else {
        return url.to_string();
    };
    // Don't redact if the `@` is past the first path slash — that's
    // RFC 3986 path syntax, not userinfo.
    if let Some(slash_offset) = rest.find('/') {
        if slash_offset < at_offset {
            return url.to_string();
        }
    }
    format!(
        "{}://<redacted>@{}",
        &url[..scheme_end],
        &rest[at_offset + 1..],
    )
}

/// Wrap a future in the optional deadline. `None` → no timeout.
async fn with_deadline<T, F>(
    deadline: Option<Duration>,
    started: Instant,
    fut: F,
) -> Result<T, BridgeError>
where
    F: std::future::Future<Output = Result<T, BridgeError>>,
{
    match deadline {
        None => fut.await,
        Some(d) => match tokio::time::timeout(d, fut).await {
            Ok(r) => r,
            Err(_) => Err(BridgeError::Timeout {
                elapsed_ms: started.elapsed().as_millis() as u64,
                cause: String::new(),
            }),
        },
    }
}

/// Map an upstream HTTP error to a customer-visible error.
///
/// **Audit-aware:** Vertex error envelopes (`{"error": {"code":
/// 403, "message": "Permission denied for project foo-bar-prod"}}`)
/// leak operator project ids. The customer-visible `message` is a
/// canned status-keyed phrase. We *do* read the upstream's
/// `error.status` (a gRPC canonical code such as `"RESOURCE_EXHAUSTED"`)
/// into [`UpstreamErrorView::kind`] so the envelope-translation layer
/// can derive an OpenAI `code` — that token is a stable taxonomy
/// label, not operator-internal data. [`UpstreamErrorView::message`]
/// is intentionally left `None`.
async fn map_http_error(status: StatusCode, resp: reqwest::Response) -> BridgeError {
    let retry_after = aisix_gateway::parse_retry_after(resp.headers());
    let is_json = aisix_gateway::response_is_json(&resp);
    let body =
        aisix_gateway::read_body_capped(resp, aisix_gateway::MAX_UPSTREAM_ERROR_BODY_BYTES).await;
    // Skip the serde parse on non-JSON bodies (HTML / text error pages
    // from a fronting WAF or load balancer). Same guard as
    // `capture_upstream_error_http`.
    let kind = is_json.then(|| parse_vertex_error_status(&body)).flatten();
    let message = match status.as_u16() {
        401 | 403 => "upstream authentication failed".to_string(),
        404 => "upstream model or endpoint not found".to_string(),
        408 => "upstream request timeout".to_string(),
        429 => "upstream rate limited".to_string(),
        _ => format!("upstream returned {}", status.as_u16()),
    };
    let parsed = kind.as_ref().map(|_| {
        Box::new(aisix_gateway::UpstreamErrorView {
            kind: kind.clone(),
            message: None,
            code: None,
            param: None,
        })
    });
    BridgeError::UpstreamStatus {
        status: status.as_u16(),
        message,
        parsed,
        wire: aisix_gateway::UpstreamWire::Vertex,
        retry_after,
    }
}

/// Extract just the `error.status` field (the gRPC canonical code) from
/// a Vertex error body. The body shape is
/// `{"error": {"code": int, "message": "...", "status": "...", "details": [...]}}`
/// per <https://cloud.google.com/apis/design/errors>. Returns `None`
/// when the body is not JSON of that shape — we intentionally do not
/// surface `error.message` (it embeds operator project ids).
fn parse_vertex_error_status(body: &[u8]) -> Option<String> {
    #[derive(serde::Deserialize)]
    struct Outer {
        error: Inner,
    }
    #[derive(serde::Deserialize)]
    struct Inner {
        status: Option<String>,
    }
    let outer: Outer = serde_json::from_slice(body).ok()?;
    outer.error.status
}

#[async_trait]
impl Bridge for VertexBridge {
    fn name(&self) -> &'static str {
        self.name
    }

    async fn chat(
        &self,
        req: &ChatFormat,
        ctx: &BridgeContext,
    ) -> Result<ChatResponse, BridgeError> {
        let upstream_id = upstream_model(ctx)?;
        let publisher = VertexPublisher::from_upstream_id(upstream_id).ok_or_else(|| {
            BridgeError::Config(format!(
                "vertex publisher unknown for upstream model id {upstream_id:?}; \
                 expected one of gemini-* / claude-* / meta/llama-* or llama* / \
                 mistral-* / jamba-*"
            ))
        })?;
        let _ = wire::reserved_query_params();

        match publisher {
            VertexPublisher::Google => self.chat_gemini(req, ctx, upstream_id).await,
            VertexPublisher::Anthropic => self.chat_anthropic(req, ctx, upstream_id).await,
            VertexPublisher::OpenAiCompat => self.chat_openai_shim(req, ctx, upstream_id).await,
            VertexPublisher::Mistral | VertexPublisher::Ai21 => {
                self.chat_mistral_ai21(req, ctx, upstream_id, publisher)
                    .await
            }
        }
    }

    async fn chat_stream(
        &self,
        req: &ChatFormat,
        ctx: &BridgeContext,
    ) -> Result<ChatChunkStream, BridgeError> {
        let upstream_id = upstream_model(ctx)?;
        let publisher = VertexPublisher::from_upstream_id(upstream_id).ok_or_else(|| {
            BridgeError::Config(format!(
                "vertex publisher unknown for upstream model id {upstream_id:?}; \
                 expected one of gemini-* / claude-* / meta/llama-* or llama* / \
                 mistral-* / jamba-*"
            ))
        })?;
        match publisher {
            VertexPublisher::Google => self.chat_gemini_stream(req, ctx, upstream_id).await,
            VertexPublisher::OpenAiCompat => {
                self.chat_openai_shim_stream(req, ctx, upstream_id).await
            }
            VertexPublisher::Anthropic => self.chat_anthropic_stream(req, ctx, upstream_id).await,
            VertexPublisher::Mistral | VertexPublisher::Ai21 => {
                self.chat_mistral_ai21_stream(req, ctx, upstream_id, publisher)
                    .await
            }
        }
    }

    /// Native Vertex embeddings (#723): google-publisher `:predict`
    /// with the text-embedding `instances` shape — LiteLLM
    /// `VertexEmbedding` (`vertex_embeddings/embedding_handler.py`)
    /// parity. Every embedding model on this adapter is a
    /// google-publisher surface (the Claude / MaaS publishers expose
    /// no embeddings), so there is no publisher dispatch here.
    async fn embed(
        &self,
        req: &EmbeddingRequest,
        ctx: &BridgeContext,
    ) -> Result<EmbeddingResponse, BridgeError> {
        let upstream_id = upstream_model(ctx)?;
        let creds = VertexSecret::parse(&ctx.provider_key.api_key)?;
        validate_url_token("project", &creds.project)?;
        validate_url_token("region", &creds.region)?;
        validate_url_token("upstream_id", upstream_id)?;

        let base = self.resolve_api_base(&creds.region, ctx.provider_key.api_base.as_deref())?;
        let url = format!(
            "{base}/v1/projects/{project}/locations/{region}/publishers/google/models/{model}:predict",
            project = creds.project,
            region = creds.region,
            model = upstream_id,
        );

        let instances: Vec<serde_json::Value> = req
            .input
            .iter()
            .map(|text| serde_json::json!({"content": text}))
            .collect();
        let mut body = serde_json::json!({"instances": instances});
        if let Some(dims) = req.dimensions {
            body["parameters"] = serde_json::json!({"outputDimensionality": dims});
        }
        apply_body_overrides(&mut body, ctx);

        let access_token = creds.resolve_access_token(&self.token_minter).await?;
        let headers = build_request_headers(
            &access_token,
            &ctx.request_id,
            ctx.provider_key.request.as_ref(),
        )?;
        let client = self.client.clone();
        let started = Instant::now();
        let model_echo = req.model.clone();

        with_deadline(ctx.deadline, started, async move {
            let resp = client
                .post(&url)
                .headers(headers)
                .json(&body)
                .send()
                .await
                .map_err(|e| BridgeError::Transport(aisix_gateway::transport_error_message(&e)))?;
            let status = resp.status();
            if !status.is_success() {
                return Err(map_http_error(status, resp).await);
            }
            let parsed: VertexPredictEmbeddingsResponse = resp
                .json()
                .await
                .map_err(|e| BridgeError::UpstreamDecode(e.to_string()))?;

            let mut data = Vec::with_capacity(parsed.predictions.len());
            let mut prompt_tokens: u64 = 0;
            for (i, p) in parsed.predictions.into_iter().enumerate() {
                if let Some(stats) = &p.embeddings.statistics {
                    prompt_tokens += stats.token_count;
                }
                data.push(EmbeddingObject {
                    index: i as u32,
                    object: "embedding".to_string(),
                    embedding: EmbeddingVector::Float(p.embeddings.values),
                });
            }
            let tokens = prompt_tokens.min(u32::MAX as u64) as u32;
            Ok(EmbeddingResponse {
                object: "list".to_string(),
                model: model_echo,
                data,
                usage: EmbeddingUsage {
                    prompt_tokens: tokens,
                    total_tokens: tokens,
                },
            })
        })
        .await
    }
}

/// `:predict` response for text-embedding models, per
/// <https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings>.
#[derive(Debug, Deserialize)]
struct VertexPredictEmbeddingsResponse {
    #[serde(default)]
    predictions: Vec<VertexEmbeddingPrediction>,
}

#[derive(Debug, Deserialize)]
struct VertexEmbeddingPrediction {
    embeddings: VertexEmbeddingsPayload,
}

#[derive(Debug, Deserialize)]
struct VertexEmbeddingsPayload {
    values: Vec<f32>,
    #[serde(default)]
    statistics: Option<VertexEmbeddingStatistics>,
}

#[derive(Debug, Deserialize)]
struct VertexEmbeddingStatistics {
    #[serde(default)]
    token_count: u64,
}

impl VertexBridge {
    /// Dispatch Gemini chat (publisher `google`). URL +
    /// body shape per
    /// <https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/gemini>.
    async fn chat_gemini(
        &self,
        req: &ChatFormat,
        ctx: &BridgeContext,
        upstream_id: &str,
    ) -> Result<ChatResponse, BridgeError> {
        let creds = VertexSecret::parse(&ctx.provider_key.api_key)?;
        // Validate all URL-path tokens to keep operator-supplied
        // strings from injecting path segments / query params.
        validate_url_token("project", &creds.project)?;
        validate_url_token("region", &creds.region)?;
        validate_url_token("upstream_id", upstream_id)?;

        let base = self.resolve_api_base(&creds.region, ctx.provider_key.api_base.as_deref())?;
        let url = format!(
            "{base}/v1/projects/{project}/locations/{region}/publishers/google/models/{model}:generateContent",
            project = creds.project,
            region = creds.region,
            model = upstream_id,
        );

        let typed = build_gemini_request(req);
        // Audit LOW-4: Gemini requires `contents` to be a non-empty
        // array. If the caller passed system-only messages (lifted to
        // `systemInstruction`), `contents` ends up empty and Vertex
        // returns a generic 400. Fail fast with a clear error so the
        // operator can fix the request shape before the round trip.
        if typed.contents.is_empty() {
            return Err(BridgeError::Config(
                "vertex chat: messages must include at least one user / \
                 assistant turn (system-only requests are not supported by Gemini)"
                    .into(),
            ));
        }
        // Serialize to JSON, then apply the per-ProviderKey override
        // pipeline (#339). The Gemini `contents` shape does not match the
        // OpenAI-style top-level keys the request transforms target, so
        // renames/clamps are usually no-ops here — but default_body_fields
        // / default_headers still apply.
        let mut body = serde_json::to_value(&typed)
            .map_err(|e| BridgeError::Config(format!("serialize Gemini request body: {e}")))?;
        apply_body_overrides(&mut body, ctx);
        // Resolve bearer: pre-minted token verbatim, or mint+cache
        // via the in-process token minter from SA JSON. Failure
        // surfaces as a Config error (operator-actionable).
        let access_token = creds.resolve_access_token(&self.token_minter).await?;
        let headers = build_request_headers(
            &access_token,
            &ctx.request_id,
            ctx.provider_key.request.as_ref(),
        )?;
        let client = self.client.clone();
        let started = Instant::now();

        with_deadline(ctx.deadline, started, async move {
            let resp = client
                .post(&url)
                .headers(headers)
                .json(&body)
                .send()
                .await
                .map_err(|e| BridgeError::Transport(aisix_gateway::transport_error_message(&e)))?;

            let status = resp.status();
            if !status.is_success() {
                return Err(map_http_error(status, resp).await);
            }
            let parsed: GeminiGenerateContentResponse = resp
                .json()
                .await
                .map_err(|e| BridgeError::UpstreamDecode(e.to_string()))?;
            Ok(gemini_response_into_chat_response(parsed, upstream_id))
        })
        .await
    }

    /// Dispatch Claude chat (publisher `anthropic`) on Vertex.
    ///
    /// URL: `<base>/v1/projects/<project>/locations/<region>/
    ///       publishers/anthropic/models/<model>:rawPredict`
    ///
    /// Unlike Gemini, Claude on Vertex speaks the **Anthropic Messages**
    /// wire — so the request body is the shared Anthropic Messages JSON
    /// (the same serializer the Bedrock `/invoke` path uses) with two
    /// Vertex-specific shaping steps per Google's Vertex AI Claude
    /// reference
    /// <https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/use-claude>
    /// and the Anthropic Vertex SDK
    /// <https://docs.anthropic.com/en/api/claude-on-vertex-ai>:
    ///
    ///   1. Strip `model` — Vertex keys the model off the URL path.
    ///   2. Strip `stream` — `:rawPredict` is the non-stream route.
    ///   3. Add `anthropic_version: "vertex-2023-10-16"` (required).
    ///
    /// Auth is the same GCP OAuth2 Bearer the Gemini path uses (minted
    /// from the SA JSON or supplied pre-minted). The response is the
    /// native Anthropic Messages envelope, decoded by the shared
    /// `response_into_chat_response`; the customer-facing alias restore
    /// happens at the proxy render layer, not here.
    async fn chat_anthropic(
        &self,
        req: &ChatFormat,
        ctx: &BridgeContext,
        upstream_id: &str,
    ) -> Result<ChatResponse, BridgeError> {
        let creds = VertexSecret::parse(&ctx.provider_key.api_key)?;
        validate_url_token("project", &creds.project)?;
        validate_url_token("region", &creds.region)?;
        validate_url_token("upstream_id", upstream_id)?;

        let base = self.resolve_api_base(&creds.region, ctx.provider_key.api_base.as_deref())?;
        let url = format!(
            "{base}/v1/projects/{project}/locations/{region}/publishers/anthropic/models/{model}:rawPredict",
            project = creds.project,
            region = creds.region,
            model = upstream_id,
        );

        // Build the Anthropic Messages body via the shared serializer,
        // then shape it for Vertex (strip model + stream, add the
        // Vertex `anthropic_version`). Mirrors the Bedrock `/invoke`
        // body shaping, differing only in the version string.
        let (system, messages) =
            split_system(req).map_err(|e| BridgeError::InvalidUpstreamConfig(format!("{e}")))?;
        let anthropic_req = build_anthropic_request(req, upstream_id, system, messages, false);
        let mut body_value = serde_json::to_value(&anthropic_req)
            .map_err(|e| BridgeError::Config(format!("serialize Anthropic request body: {e}")))?;
        // Apply the per-ProviderKey override pipeline (#339) before the
        // Vertex-specific shaping below, so the `model`/`stream` strip keeps
        // the final say and an override can never reintroduce a URL-borne
        // `model` into the `:rawPredict` body.
        apply_body_overrides(&mut body_value, ctx);
        if let Some(obj) = body_value.as_object_mut() {
            obj.remove("model");
            obj.remove("stream");
            obj.insert(
                "anthropic_version".to_string(),
                serde_json::Value::String(VERTEX_ANTHROPIC_VERSION.to_string()),
            );
        }

        let access_token = creds.resolve_access_token(&self.token_minter).await?;
        let headers = build_request_headers(
            &access_token,
            &ctx.request_id,
            ctx.provider_key.request.as_ref(),
        )?;
        let client = self.client.clone();
        let started = Instant::now();

        with_deadline(ctx.deadline, started, async move {
            let resp = client
                .post(&url)
                .headers(headers)
                .json(&body_value)
                .send()
                .await
                .map_err(|e| BridgeError::Transport(aisix_gateway::transport_error_message(&e)))?;

            let status = resp.status();
            if !status.is_success() {
                return Err(map_http_error(status, resp).await);
            }
            let parsed: AnthropicResponse = resp
                .json()
                .await
                .map_err(|e| BridgeError::UpstreamDecode(e.to_string()))?;
            Ok(response_into_chat_response(parsed))
        })
        .await
    }

    /// Streaming counterpart of [`Self::chat_anthropic`]. Claude on
    /// Vertex streams via the `:streamRawPredict` action with the SAME
    /// Anthropic Messages body shaping as the non-stream path EXCEPT
    /// `stream: true` is KEPT in the body. The URL action alone does not
    /// signal streaming: the Anthropic Vertex SDK pops `model` into the
    /// URL but *reads* (not removes) `stream`, so `"stream": true` still
    /// rides in the body
    /// <https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/lib/vertex/_client.py>.
    ///
    /// The response is native Anthropic SSE (`message_start` /
    /// `content_block_delta` / `message_delta` / `message_stop`), decoded
    /// by the shared [`AnthropicStreamEvent`] + [`StreamState`] — the
    /// exact decoder the direct Anthropic bridge drives, so streaming
    /// behaviour matches direct Anthropic. Auth + the customer-facing
    /// alias restore are identical to the non-stream path.
    async fn chat_anthropic_stream(
        &self,
        req: &ChatFormat,
        ctx: &BridgeContext,
        upstream_id: &str,
    ) -> Result<ChatChunkStream, BridgeError> {
        let creds = VertexSecret::parse(&ctx.provider_key.api_key)?;
        validate_url_token("project", &creds.project)?;
        validate_url_token("region", &creds.region)?;
        validate_url_token("upstream_id", upstream_id)?;

        let base = self.resolve_api_base(&creds.region, ctx.provider_key.api_base.as_deref())?;
        let url = format!(
            "{base}/v1/projects/{project}/locations/{region}/publishers/anthropic/models/{model}:streamRawPredict",
            project = creds.project,
            region = creds.region,
            model = upstream_id,
        );

        // Same Anthropic Messages body as the non-stream path, but built
        // with stream=true and — unlike `:rawPredict` — `stream` is KEPT
        // in the body (only `model` is stripped into the URL). Add the
        // Vertex `anthropic_version`.
        let (system, messages) =
            split_system(req).map_err(|e| BridgeError::InvalidUpstreamConfig(format!("{e}")))?;
        let anthropic_req = build_anthropic_request(req, upstream_id, system, messages, true);
        let mut body_value = serde_json::to_value(&anthropic_req)
            .map_err(|e| BridgeError::Config(format!("serialize Anthropic request body: {e}")))?;
        // Apply the per-ProviderKey override pipeline (#339) before the
        // Vertex-specific shaping below (see the non-stream path). Here
        // `stream` is intentionally KEPT in the body.
        apply_body_overrides(&mut body_value, ctx);
        if let Some(obj) = body_value.as_object_mut() {
            obj.remove("model");
            obj.insert(
                "anthropic_version".to_string(),
                serde_json::Value::String(VERTEX_ANTHROPIC_VERSION.to_string()),
            );
        }

        // Resolve bearer BEFORE entering the stream future so a
        // token-mint error surfaces as a direct Err, not mid-stream.
        let access_token = creds.resolve_access_token(&self.token_minter).await?;
        let headers = build_request_headers(
            &access_token,
            &ctx.request_id,
            ctx.provider_key.request.as_ref(),
        )?;
        let client = self.client.clone();
        let started = Instant::now();

        let resp = with_deadline(ctx.deadline, started, async move {
            client
                .post(&url)
                .headers(headers)
                .json(&body_value)
                .send()
                .await
                .map_err(|e| BridgeError::Transport(aisix_gateway::transport_error_message(&e)))
        })
        .await?;

        let status = resp.status();
        if !status.is_success() {
            return Err(map_http_error(status, resp).await);
        }

        // Native Anthropic SSE → ChatChunk, driven by the shared
        // StreamState. Mirrors aisix-provider-anthropic's
        // `build_chunk_stream`: parse each `data:` payload into an
        // AnthropicStreamEvent, update rolling state (id/model carried
        // from `message_start`), emit any chunk, and stop on the
        // terminal `message_stop`. Anthropic emits no `[DONE]` sentinel.
        let byte_stream = resp.bytes_stream();
        let stream = async_stream::try_stream! {
            let mut decoder = SseDecoder::new();
            let mut state = StreamState::default();
            let mut byte_stream = Box::pin(byte_stream);

            while let Some(item) = byte_stream.next().await {
                let bytes: Bytes = item.map_err(|e| BridgeError::Transport(aisix_gateway::transport_error_message(&e)))?;
                for event in decoder.feed(bytes.as_ref()) {
                    let SseEvent::Data(data) = event else { continue };
                    let parsed: AnthropicStreamEvent =
                        serde_json::from_str(&data).map_err(|e| {
                            BridgeError::UpstreamDecode(format!(
                                "vertex anthropic stream chunk parse: {e}"
                            ))
                        })?;
                    state.update(&parsed);
                    if let Some(chunk) = state.to_chunk(&parsed) {
                        yield chunk;
                    }
                    if StreamState::is_terminal(&parsed) {
                        return;
                    }
                }
            }
        };

        Ok(Box::pin(stream))
    }

    /// Build the OpenAI-shim chat-completions URL for the MaaS family.
    /// Unlike the Gemini / Anthropic `publishers/<vendor>/...` paths,
    /// the shim has NO publisher segment and NO model in the path — the
    /// model rides in the request body. Only `project` + `region` are
    /// URL-path tokens (validated by the caller).
    fn openai_shim_url(
        &self,
        creds: &VertexSecret,
        api_base: Option<&str>,
    ) -> Result<String, BridgeError> {
        let base = self.resolve_api_base(&creds.region, api_base)?;
        Ok(format!(
            "{base}/v1/projects/{project}/locations/{region}/endpoints/openapi/chat/completions",
            project = creds.project,
            region = creds.region,
        ))
    }

    /// Dispatch Llama + the OpenAI-compatible MaaS family (DeepSeek /
    /// Qwen / gpt-oss / MiniMax / Moonshot / Z.ai) non-stream chat via
    /// Vertex's OpenAI chat-completions shim. Body + response are the
    /// OpenAI wire, reused verbatim from the OpenAI bridge's serializer
    /// / decoder, so behaviour matches direct OpenAI. The model id is
    /// kept in the body's `model` field (the shim keys off it, not the
    /// URL). Reference impl confirms the shim endpoint + OpenAI body:
    /// <https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/llama#openai>.
    async fn chat_openai_shim(
        &self,
        req: &ChatFormat,
        ctx: &BridgeContext,
        upstream_id: &str,
    ) -> Result<ChatResponse, BridgeError> {
        let creds = VertexSecret::parse(&ctx.provider_key.api_key)?;
        // Only project + region are URL-path tokens; the model id rides
        // in the body and may legitimately contain `/` (e.g.
        // `meta/llama-3.3-70b-instruct-maas`), so it is NOT validated as
        // a URL token here.
        validate_url_token("project", &creds.project)?;
        validate_url_token("region", &creds.region)?;

        let url = self.openai_shim_url(&creds, ctx.provider_key.api_base.as_deref())?;

        let messages = openai_messages_from(req);
        let typed = build_openai_request(req, upstream_id, &messages, false);
        let mut body = serde_json::to_value(&typed)
            .map_err(|e| BridgeError::Config(format!("serialize OpenAI shim request body: {e}")))?;
        // Apply the per-ProviderKey override pipeline (#339). The shim
        // speaks the OpenAI wire, so renames / clamps / default fields all
        // apply directly; the model id is kept in the body.
        apply_body_overrides(&mut body, ctx);

        let access_token = creds.resolve_access_token(&self.token_minter).await?;
        let headers = build_request_headers(
            &access_token,
            &ctx.request_id,
            ctx.provider_key.request.as_ref(),
        )?;
        let client = self.client.clone();
        let started = Instant::now();

        with_deadline(ctx.deadline, started, async move {
            let resp = client
                .post(&url)
                .headers(headers)
                .json(&body)
                .send()
                .await
                .map_err(|e| BridgeError::Transport(aisix_gateway::transport_error_message(&e)))?;
            let status = resp.status();
            if !status.is_success() {
                return Err(map_http_error(status, resp).await);
            }
            let parsed: OpenAiResponse = resp
                .json()
                .await
                .map_err(|e| BridgeError::UpstreamDecode(e.to_string()))?;
            Ok(openai_response_into_chat_response(parsed))
        })
        .await
    }

    /// Streaming counterpart of [`Self::chat_openai_shim`]. The shim
    /// emits OpenAI-style SSE (`data: {chunk}` lines terminated by
    /// `data: [DONE]`), so reuse the shared [`SseDecoder`] + the OpenAI
    /// stream-chunk decoder. `stream: true` goes in the body (the shim
    /// has no `?alt=sse`-style query).
    async fn chat_openai_shim_stream(
        &self,
        req: &ChatFormat,
        ctx: &BridgeContext,
        upstream_id: &str,
    ) -> Result<ChatChunkStream, BridgeError> {
        let creds = VertexSecret::parse(&ctx.provider_key.api_key)?;
        validate_url_token("project", &creds.project)?;
        validate_url_token("region", &creds.region)?;

        let url = self.openai_shim_url(&creds, ctx.provider_key.api_base.as_deref())?;

        let messages = openai_messages_from(req);
        let typed = build_openai_request(req, upstream_id, &messages, true);
        let mut body = serde_json::to_value(&typed)
            .map_err(|e| BridgeError::Config(format!("serialize OpenAI shim request body: {e}")))?;
        // Apply the per-ProviderKey override pipeline (#339); `stream: true`
        // stays in the body.
        apply_body_overrides(&mut body, ctx);

        // Resolve bearer BEFORE entering the stream future so a
        // token-mint error surfaces as a direct Err, not mid-stream.
        let access_token = creds.resolve_access_token(&self.token_minter).await?;
        let headers = build_request_headers(
            &access_token,
            &ctx.request_id,
            ctx.provider_key.request.as_ref(),
        )?;
        let client = self.client.clone();
        let started = Instant::now();

        let resp = with_deadline(ctx.deadline, started, async move {
            client
                .post(&url)
                .headers(headers)
                .json(&body)
                .send()
                .await
                .map_err(|e| BridgeError::Transport(aisix_gateway::transport_error_message(&e)))
        })
        .await?;

        let status = resp.status();
        if !status.is_success() {
            return Err(map_http_error(status, resp).await);
        }

        let byte_stream = resp.bytes_stream();
        let stream = async_stream::try_stream! {
            let mut decoder = SseDecoder::new();
            let mut byte_stream = Box::pin(byte_stream);

            while let Some(item) = byte_stream.next().await {
                let bytes: Bytes = item.map_err(|e| BridgeError::Transport(aisix_gateway::transport_error_message(&e)))?;
                for event in decoder.feed(bytes.as_ref()) {
                    match event {
                        SseEvent::Data(data) => {
                            let parsed: OpenAiStreamChunk =
                                serde_json::from_str(&data).map_err(|e| {
                                    BridgeError::UpstreamDecode(format!(
                                        "vertex openai-shim stream chunk parse: {e}"
                                    ))
                                })?;
                            yield openai_stream_chunk_into_chat_chunk(parsed);
                        }
                        // OpenAI shim terminates with `data: [DONE]`;
                        // stop emitting on the sentinel.
                        SseEvent::Done => {}
                    }
                }
            }
            // Flush a partial trailing chunk if the connection drops
            // without a final blank line.
            if let Some(SseEvent::Data(data)) = decoder.finish() {
                let parsed: OpenAiStreamChunk = serde_json::from_str(&data).map_err(|e| {
                    BridgeError::UpstreamDecode(format!(
                        "vertex openai-shim stream tail parse: {e}"
                    ))
                })?;
                yield openai_stream_chunk_into_chat_chunk(parsed);
            }
        };

        Ok(Box::pin(stream))
    }

    /// Build the `:rawPredict` / `:streamRawPredict` URL for the Mistral
    /// (`publishers/mistralai`) and AI21 (`publishers/ai21`) partner rail.
    /// UNLIKE the OpenAI shim (`endpoints/openapi/chat/completions`, no
    /// model in the URL), these partners use a
    /// `publishers/<vendor>/models/<model>:<action>` URL AND keep the
    /// model in the OpenAI body — the model id appears in BOTH. `segment`
    /// is `publisher.url_segment()` (`publishers/mistralai` |
    /// `publishers/ai21`); `action` is `rawPredict` | `streamRawPredict`.
    fn partner_rawpredict_url(
        &self,
        creds: &VertexSecret,
        api_base: Option<&str>,
        segment: &str,
        model: &str,
        action: &str,
    ) -> Result<String, BridgeError> {
        let base = self.resolve_api_base(&creds.region, api_base)?;
        Ok(format!(
            "{base}/v1/projects/{project}/locations/{region}/{segment}/models/{model}:{action}",
            project = creds.project,
            region = creds.region,
        ))
    }

    /// Dispatch Mistral (`mistral-*` / `codestral-*`) and AI21 Jamba
    /// (`jamba-*`) non-stream chat. Both speak an OpenAI-compatible body
    /// over the partner `:rawPredict` URL — so reuse the OpenAI bridge's
    /// serializer + response decoder verbatim. UNLIKE the OpenAI shim, the
    /// model id is a URL path segment here (and is ALSO kept in the body),
    /// so validate it as a URL token. Wire confirmed against the Vertex
    /// Mistral docs
    /// <https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/mistral>
    /// and Google's Vertex AI21 sample (GoogleCloudPlatform/vertex-ai-samples
    /// `ai21labs_intro.ipynb`): a `{model, messages, max_tokens, stream}`
    /// body POSTed to `publishers/{mistralai|ai21}/models/<model>:rawPredict`.
    async fn chat_mistral_ai21(
        &self,
        req: &ChatFormat,
        ctx: &BridgeContext,
        upstream_id: &str,
        publisher: VertexPublisher,
    ) -> Result<ChatResponse, BridgeError> {
        let creds = VertexSecret::parse(&ctx.provider_key.api_key)?;
        validate_url_token("project", &creds.project)?;
        validate_url_token("region", &creds.region)?;
        // The model id is a URL path segment here (Mistral / AI21 ids are
        // clean — no `/`, no `@` — so this never rejects a valid id).
        validate_url_token("upstream_id", upstream_id)?;

        let segment = publisher.url_segment().ok_or_else(|| {
            BridgeError::Config(format!(
                "vertex publisher {publisher:?} has no URL segment for the :rawPredict rail"
            ))
        })?;
        let url = self.partner_rawpredict_url(
            &creds,
            ctx.provider_key.api_base.as_deref(),
            segment,
            upstream_id,
            "rawPredict",
        )?;

        // OpenAI chat-completions body — same serializer the OpenAI-shim
        // (Llama/MaaS) rail uses. The model is KEPT in the body (Mistral /
        // AI21 on Vertex expect it in both the URL and the body).
        let messages = openai_messages_from(req);
        let typed = build_openai_request(req, upstream_id, &messages, false);
        let mut body = serde_json::to_value(&typed)
            .map_err(|e| BridgeError::Config(format!("serialize OpenAI request body: {e}")))?;
        // Apply the per-ProviderKey override pipeline (#339). The model id
        // is kept in the body (Mistral / AI21 expect it in both URL + body).
        apply_body_overrides(&mut body, ctx);

        let access_token = creds.resolve_access_token(&self.token_minter).await?;
        let headers = build_request_headers(
            &access_token,
            &ctx.request_id,
            ctx.provider_key.request.as_ref(),
        )?;
        let client = self.client.clone();
        let started = Instant::now();

        with_deadline(ctx.deadline, started, async move {
            let resp = client
                .post(&url)
                .headers(headers)
                .json(&body)
                .send()
                .await
                .map_err(|e| BridgeError::Transport(aisix_gateway::transport_error_message(&e)))?;
            let status = resp.status();
            if !status.is_success() {
                return Err(map_http_error(status, resp).await);
            }
            let parsed: OpenAiResponse = resp
                .json()
                .await
                .map_err(|e| BridgeError::UpstreamDecode(e.to_string()))?;
            Ok(openai_response_into_chat_response(parsed))
        })
        .await
    }

    /// Streaming counterpart of [`Self::chat_mistral_ai21`]. Same OpenAI
    /// body + decoders as the non-stream path, but the `:streamRawPredict`
    /// action and `stream: true` kept in the body; the response is OpenAI
    /// SSE (`data: {chunk}` terminated by `data: [DONE]`), decoded by the
    /// shared [`SseDecoder`] + the OpenAI stream-chunk decoder.
    async fn chat_mistral_ai21_stream(
        &self,
        req: &ChatFormat,
        ctx: &BridgeContext,
        upstream_id: &str,
        publisher: VertexPublisher,
    ) -> Result<ChatChunkStream, BridgeError> {
        let creds = VertexSecret::parse(&ctx.provider_key.api_key)?;
        validate_url_token("project", &creds.project)?;
        validate_url_token("region", &creds.region)?;
        validate_url_token("upstream_id", upstream_id)?;

        let segment = publisher.url_segment().ok_or_else(|| {
            BridgeError::Config(format!(
                "vertex publisher {publisher:?} has no URL segment for the :streamRawPredict rail"
            ))
        })?;
        let url = self.partner_rawpredict_url(
            &creds,
            ctx.provider_key.api_base.as_deref(),
            segment,
            upstream_id,
            "streamRawPredict",
        )?;

        let messages = openai_messages_from(req);
        let typed = build_openai_request(req, upstream_id, &messages, true);
        let mut body = serde_json::to_value(&typed)
            .map_err(|e| BridgeError::Config(format!("serialize OpenAI request body: {e}")))?;
        // Apply the per-ProviderKey override pipeline (#339); `stream: true`
        // stays in the body (model id rides in both URL + body).
        apply_body_overrides(&mut body, ctx);

        // Resolve bearer BEFORE entering the stream future so a
        // token-mint error surfaces as a direct Err, not mid-stream.
        let access_token = creds.resolve_access_token(&self.token_minter).await?;
        let headers = build_request_headers(
            &access_token,
            &ctx.request_id,
            ctx.provider_key.request.as_ref(),
        )?;
        let client = self.client.clone();
        let started = Instant::now();

        let resp = with_deadline(ctx.deadline, started, async move {
            client
                .post(&url)
                .headers(headers)
                .json(&body)
                .send()
                .await
                .map_err(|e| BridgeError::Transport(aisix_gateway::transport_error_message(&e)))
        })
        .await?;

        let status = resp.status();
        if !status.is_success() {
            return Err(map_http_error(status, resp).await);
        }

        let byte_stream = resp.bytes_stream();
        let stream = async_stream::try_stream! {
            let mut decoder = SseDecoder::new();
            let mut byte_stream = Box::pin(byte_stream);

            while let Some(item) = byte_stream.next().await {
                let bytes: Bytes = item.map_err(|e| BridgeError::Transport(aisix_gateway::transport_error_message(&e)))?;
                for event in decoder.feed(bytes.as_ref()) {
                    match event {
                        SseEvent::Data(data) => {
                            let parsed: OpenAiStreamChunk =
                                serde_json::from_str(&data).map_err(|e| {
                                    BridgeError::UpstreamDecode(format!(
                                        "vertex partner :streamRawPredict chunk parse: {e}"
                                    ))
                                })?;
                            yield openai_stream_chunk_into_chat_chunk(parsed);
                        }
                        // OpenAI-compatible upstreams (Mistral / AI21)
                        // terminate with `data: [DONE]`; stop on the sentinel.
                        SseEvent::Done => {}
                    }
                }
            }
            if let Some(SseEvent::Data(data)) = decoder.finish() {
                let parsed: OpenAiStreamChunk = serde_json::from_str(&data).map_err(|e| {
                    BridgeError::UpstreamDecode(format!(
                        "vertex partner :streamRawPredict tail parse: {e}"
                    ))
                })?;
                yield openai_stream_chunk_into_chat_chunk(parsed);
            }
        };

        Ok(Box::pin(stream))
    }

    /// Dispatch Gemini streaming chat (publisher `google`).
    ///
    /// URL: `<base>/v1/projects/<project>/locations/<region>/
    ///       publishers/google/models/<model>:streamGenerateContent?alt=sse`
    ///
    /// Per the official Vertex AI REST docs
    /// <https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference#stream>,
    /// the `?alt=sse` query selects the SSE wire (`data: {json}\n\n`
    /// chunks). Without it the same path emits a JSON array stream
    /// (one JSON object per chunk, comma-separated) — not implemented
    /// here because every DP-side consumer wants SSE.
    ///
    /// Unlike OpenAI, Gemini does NOT emit a `data: [DONE]` sentinel
    /// — the upstream simply closes the connection after the last
    /// chunk (typically the one carrying `finishReason` and
    /// `usageMetadata`). The decoder treats either pattern as end of
    /// stream so a future upstream change that adds `[DONE]` doesn't
    /// break us.
    async fn chat_gemini_stream(
        &self,
        req: &ChatFormat,
        ctx: &BridgeContext,
        upstream_id: &str,
    ) -> Result<ChatChunkStream, BridgeError> {
        let creds = VertexSecret::parse(&ctx.provider_key.api_key)?;
        validate_url_token("project", &creds.project)?;
        validate_url_token("region", &creds.region)?;
        validate_url_token("upstream_id", upstream_id)?;

        let base = self.resolve_api_base(&creds.region, ctx.provider_key.api_base.as_deref())?;
        let url = format!(
            "{base}/v1/projects/{project}/locations/{region}/publishers/google/models/{model}:streamGenerateContent?alt=sse",
            project = creds.project,
            region = creds.region,
            model = upstream_id,
        );

        let typed = build_gemini_request(req);
        if typed.contents.is_empty() {
            return Err(BridgeError::Config(
                "vertex chat: messages must include at least one user / \
                 assistant turn (system-only requests are not supported by Gemini)"
                    .into(),
            ));
        }
        // Serialize + apply the per-ProviderKey override pipeline (#339)
        // before sending (see the non-stream path for the rail caveat).
        let mut body = serde_json::to_value(&typed)
            .map_err(|e| BridgeError::Config(format!("serialize Gemini request body: {e}")))?;
        apply_body_overrides(&mut body, ctx);
        // Resolve bearer (pre-minted OR minted-from-SA) BEFORE
        // entering the stream future so token-mint errors surface
        // as a direct Err return rather than being yielded mid-stream.
        let access_token = creds.resolve_access_token(&self.token_minter).await?;
        let headers = build_request_headers(
            &access_token,
            &ctx.request_id,
            ctx.provider_key.request.as_ref(),
        )?;
        let client = self.client.clone();
        let started = Instant::now();

        let resp = with_deadline(ctx.deadline, started, async move {
            client
                .post(&url)
                .headers(headers)
                .json(&body)
                .send()
                .await
                .map_err(|e| BridgeError::Transport(aisix_gateway::transport_error_message(&e)))
        })
        .await?;

        let status = resp.status();
        if !status.is_success() {
            return Err(map_http_error(status, resp).await);
        }

        let upstream_id_owned = upstream_id.to_string();
        let byte_stream = resp.bytes_stream();

        // try_stream! adapts the byte stream into a Stream of
        // Result<ChatChunk, BridgeError>. Pattern mirrors the
        // openai bridge's chat_stream — see
        // `aisix-provider-openai::build_chunk_stream`.
        let stream = async_stream::try_stream! {
            let mut decoder = SseDecoder::new();
            let mut emitted_role = false;
            let mut byte_stream = Box::pin(byte_stream);

            while let Some(item) = byte_stream.next().await {
                let bytes: Bytes = item.map_err(|e| BridgeError::Transport(aisix_gateway::transport_error_message(&e)))?;
                for event in decoder.feed(bytes.as_ref()) {
                    if let SseEvent::Data(data) = event {
                        let parsed: GeminiGenerateContentResponse =
                            serde_json::from_str(&data).map_err(|e| {
                                BridgeError::UpstreamDecode(format!(
                                    "vertex stream chunk parse: {e}"
                                ))
                            })?;
                        for chunk in
                            gemini_chunk_into_chat_chunks(parsed, &upstream_id_owned, &mut emitted_role)
                        {
                            yield chunk;
                        }
                    }
                    // SseEvent::Done would fire only if upstream emits
                    // [DONE]; Gemini doesn't. We tolerate either pattern
                    // by simply ignoring the sentinel — the stream-close
                    // signal is the byte stream ending.
                }
            }
            // Flush any buffered trailing bytes — Gemini closes
            // cleanly so this rarely fires, but it covers a partial
            // last chunk if the upstream connection drops without a
            // final `\n\n`.
            if let Some(SseEvent::Data(data)) = decoder.finish() {
                let parsed: GeminiGenerateContentResponse =
                    serde_json::from_str(&data).map_err(|e| {
                        BridgeError::UpstreamDecode(format!(
                            "vertex stream tail parse: {e}"
                        ))
                    })?;
                for chunk in
                    gemini_chunk_into_chat_chunks(parsed, &upstream_id_owned, &mut emitted_role)
                {
                    yield chunk;
                }
            }
        };

        Ok(Box::pin(stream))
    }
}

/// Translate one Gemini stream chunk into the gateway's
/// [`ChatChunk`] sequence. Returns up to TWO chunks per upstream
/// chunk:
///
/// 1. **Role chunk** (only on the first emission carrying text) —
///    sets `delta.role = Role::Assistant` so downstream consumers
///    know this is an assistant turn. Mirrors OpenAI's
///    "first-chunk-is-role" wire convention.
/// 2. **Content / terminal chunk** — carries the text delta and,
///    on the final upstream chunk, the `finish_reason` +
///    `usage` populated from `finishReason` + `usageMetadata`.
///
/// Returns 0 chunks if the upstream chunk had no candidates AND no
/// usage metadata (which would be an empty keep-alive — Gemini
/// doesn't emit those today, but the no-op is the safe response).
fn gemini_chunk_into_chat_chunks(
    raw: GeminiGenerateContentResponse,
    upstream_id: &str,
    emitted_role: &mut bool,
) -> Vec<ChatChunk> {
    let first_candidate = raw.candidates.into_iter().next();
    let (text, finish_reason_raw) = match first_candidate {
        Some(c) => {
            let text = c
                .content
                .map(|ct| {
                    ct.parts
                        .into_iter()
                        .filter_map(|p| p.text)
                        .collect::<Vec<_>>()
                        .join("")
                })
                .unwrap_or_default();
            (text, c.finish_reason)
        }
        None => (String::new(), None),
    };
    // Distinguish "no finishReason field" from "finishReason: STOP".
    // The non-stream path collapses both to FinishReason::Stop (since
    // ChatResponse always carries a finish_reason); on the stream
    // path we only attach finish_reason when upstream explicitly set
    // one, otherwise downstream consumers assume "stream still going".
    let finish_reason = finish_reason_raw
        .as_deref()
        .map(|s| map_gemini_finish_reason(Some(s)));

    let usage = raw.usage_metadata.map(|u| UsageStats {
        prompt_tokens: u.prompt_token_count,
        completion_tokens: u.candidates_token_count,
        total_tokens: if u.total_token_count > 0 {
            u.total_token_count
        } else {
            u.prompt_token_count
                .saturating_add(u.candidates_token_count)
        },
        ..Default::default()
    });

    let mut chunks = Vec::with_capacity(2);

    // Role-setting chunk on first content emission.
    if !*emitted_role && !text.is_empty() {
        *emitted_role = true;
        chunks.push(ChatChunk {
            id: String::new(),
            model: upstream_id.to_string(),
            delta: ChatDelta {
                role: Some(Role::Assistant),
                ..Default::default()
            },
            finish_reason: None,
            usage: None,
        });
    }

    // Content / terminal chunk — emit when there's text OR a
    // finishReason OR usage, so the terminal-state chunk lands even
    // when its content is empty.
    if !text.is_empty() || finish_reason.is_some() || usage.is_some() {
        chunks.push(ChatChunk {
            id: String::new(),
            model: upstream_id.to_string(),
            delta: ChatDelta {
                content: (!text.is_empty()).then_some(text),
                ..Default::default()
            },
            finish_reason,
            usage,
        });
    }

    chunks
}

/// Build the outbound headers: `Authorization: Bearer <access_token>`,
/// `Content-Type: application/json`, `x-aisix-request-id`. The Bearer
/// token is the pre-minted GCP OAuth2 access token.
///
/// **Audit MEDIUM-1:** header-invalid errors deliberately drop the
/// underlying `InvalidHeaderValue` Display output. The `http` crate's
/// current Display impl is opaque, but it's an implementation detail
/// — a future change could include the offending byte position, which
/// for `access_token` would leak partial secret content. The bytes
/// being validated ARE the customer's bearer token; the operator can
/// reproduce locally without us echoing them back.
///
/// When the `ProviderKey` carries `request.default_headers`, they are
/// applied last via [`apply_default_headers`], which refuses to overwrite
/// any header already set here (the Bearer auth, content-type, and request
/// id) and additionally drops reserved auth headers — so a misconfigured
/// `default_headers` block can never clobber the Vertex OAuth Bearer.
fn build_request_headers(
    access_token: &str,
    request_id: &str,
    request: Option<&RequestOverrides>,
) -> Result<HeaderMap, BridgeError> {
    if access_token.is_empty() {
        return Err(BridgeError::Config(
            "vertex provider_key.api_key.access_token is empty".into(),
        ));
    }
    let mut headers = HeaderMap::new();
    let auth = HeaderValue::from_str(&format!("Bearer {access_token}")).map_err(|_| {
        BridgeError::Config("access_token contains invalid header characters".into())
    })?;
    headers.insert(header::AUTHORIZATION, auth);
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    let rid = HeaderValue::from_str(request_id)
        .map_err(|_| BridgeError::Config("request_id contains invalid header characters".into()))?;
    headers.insert(HeaderName::from_static("x-aisix-request-id"), rid);
    if let Some(r) = request {
        apply_default_headers(&mut headers, &r.default_headers);
    }
    Ok(headers)
}

/// Apply the per-`ProviderKey` request/response override pipeline to an
/// already-serialized outbound body, mirroring
/// `OpenAiBridge::prepare_outbound_body` step-for-step:
/// `param_renames` → `param_constraints` → `default_body_fields` →
/// (`content_list_to_string`, when the response override sets it).
///
/// Called on every publisher rail immediately after the wire body is
/// serialized to a [`serde_json::Value`] and **before** any Vertex-specific
/// shaping (e.g. the Anthropic `model`/`stream` strip), so the rail-invariant
/// shaping always has the final say and the override block can never
/// reintroduce a URL-borne `model` into a `:rawPredict` body.
fn apply_body_overrides(body: &mut serde_json::Value, ctx: &BridgeContext) {
    if let Some(r) = ctx.provider_key.request.as_ref() {
        apply_param_renames(body, &r.param_renames);
        if let Some(constraints) = &r.param_constraints {
            apply_param_constraints(body, constraints);
        }
        apply_default_body_fields(body, &r.default_body_fields);
    }
    if ctx
        .provider_key
        .response
        .as_ref()
        .is_some_and(|r| r.content_list_to_string)
    {
        apply_content_list_to_string(body);
    }
}

// ─── Gemini wire shapes ────────────────────────────────────────────────

/// Gemini's `generateContent` request body per
/// <https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/gemini>.
///
/// Note `system_instruction` is OPTIONAL and only emitted when
/// the caller's ChatFormat has system-role turns; sending an empty
/// one would 400 upstream. Same goes for `generation_config`.
#[derive(Debug, Serialize)]
struct GeminiGenerateContentRequest {
    contents: Vec<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "systemInstruction")]
    system_instruction: Option<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "generationConfig")]
    generation_config: Option<GeminiGenerationConfig>,
}

#[derive(Debug, Serialize)]
struct GeminiContent {
    /// Gemini accepts `"user"` and `"model"` (no `"assistant"`).
    /// System messages are NOT in `contents` — they go in the
    /// top-level `systemInstruction` field.
    role: &'static str,
    parts: Vec<GeminiPart>,
}

#[derive(Debug, Serialize)]
struct GeminiPart {
    /// Single text part. Vision / multimodal parts (`inlineData`,
    /// `fileData`) deferred to a follow-up.
    text: String,
}

#[derive(Debug, Serialize, Default)]
struct GeminiGenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "topP")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "maxOutputTokens")]
    max_output_tokens: Option<u32>,
}

/// Translate the gateway's [`ChatFormat`] into Gemini's
/// `generateContent` body.
///
/// Translation rules:
/// - System messages → top-level `systemInstruction` (concatenated
///   with `\n\n` if multiple). They do NOT appear in `contents`.
/// - User messages → `{"role":"user","parts":[{"text":...}]}`
/// - Assistant messages → `{"role":"model","parts":[{"text":...}]}`
///   (Gemini uses `"model"` not `"assistant"`)
/// - Tool messages: out of scope for D5.2.a; treated as user text
///   (preserves conversation history without 400ing the upstream)
/// - `temperature`, `top_p`, `max_tokens` → `generationConfig.*`
fn build_gemini_request(req: &ChatFormat) -> GeminiGenerateContentRequest {
    let mut system_parts: Vec<String> = Vec::new();
    let mut contents: Vec<GeminiContent> = Vec::new();
    for m in &req.messages {
        match m.role {
            Role::System => system_parts.push(m.content_str().to_string()),
            Role::User | Role::Tool => contents.push(GeminiContent {
                role: "user",
                parts: vec![GeminiPart {
                    text: m.content_str().to_string(),
                }],
            }),
            Role::Assistant => contents.push(GeminiContent {
                role: "model",
                parts: vec![GeminiPart {
                    text: m.content_str().to_string(),
                }],
            }),
        }
    }
    let system_instruction = if system_parts.is_empty() {
        None
    } else {
        Some(GeminiContent {
            role: "user", // Gemini ignores `role` inside systemInstruction; "user" is the convention.
            parts: vec![GeminiPart {
                text: system_parts.join("\n\n"),
            }],
        })
    };
    let generation_config =
        if req.temperature.is_some() || req.top_p.is_some() || req.max_tokens.is_some() {
            Some(GeminiGenerationConfig {
                temperature: req.temperature,
                top_p: req.top_p,
                max_output_tokens: req.max_tokens,
            })
        } else {
            None
        };
    GeminiGenerateContentRequest {
        contents,
        system_instruction,
        generation_config,
    }
}

/// Gemini's `generateContent` response shape.
#[derive(Debug, Deserialize)]
struct GeminiGenerateContentResponse {
    #[serde(default)]
    candidates: Vec<GeminiCandidate>,
    #[serde(default, rename = "usageMetadata")]
    usage_metadata: Option<GeminiUsageMetadata>,
}

#[derive(Debug, Deserialize)]
struct GeminiCandidate {
    content: Option<GeminiResponseContent>,
    #[serde(default, rename = "finishReason")]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GeminiResponseContent {
    #[serde(default)]
    parts: Vec<GeminiResponsePart>,
    // role is always "model" — ignored.
}

#[derive(Debug, Deserialize)]
struct GeminiResponsePart {
    #[serde(default)]
    text: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct GeminiUsageMetadata {
    #[serde(default, rename = "promptTokenCount")]
    prompt_token_count: u32,
    #[serde(default, rename = "candidatesTokenCount")]
    candidates_token_count: u32,
    #[serde(default, rename = "totalTokenCount")]
    total_token_count: u32,
}

/// Translate Gemini's response into the gateway's [`ChatResponse`].
fn gemini_response_into_chat_response(
    raw: GeminiGenerateContentResponse,
    upstream_id: &str,
) -> ChatResponse {
    let first = raw.candidates.into_iter().next();
    let (message, finish) = match first {
        Some(c) => {
            let text: String = c
                .content
                .map(|ct| {
                    ct.parts
                        .into_iter()
                        .filter_map(|p| p.text)
                        .collect::<Vec<_>>()
                        .join("")
                })
                .unwrap_or_default();
            (
                ChatMessage::assistant(text),
                map_gemini_finish_reason(c.finish_reason.as_deref()),
            )
        }
        None => (ChatMessage::assistant(""), FinishReason::Stop),
    };
    let usage = raw
        .usage_metadata
        .map(|u| UsageStats {
            prompt_tokens: u.prompt_token_count,
            completion_tokens: u.candidates_token_count,
            total_tokens: if u.total_token_count > 0 {
                u.total_token_count
            } else {
                u.prompt_token_count
                    .saturating_add(u.candidates_token_count)
            },
            ..Default::default()
        })
        .unwrap_or_default();
    ChatResponse {
        id: String::new(), // Gemini doesn't return a request id in the body
        model: upstream_id.to_string(),
        message,
        finish_reason: finish,
        usage,
    }
}

/// Map Gemini's `finishReason` strings to the gateway's enum. Per
/// <https://ai.google.dev/api/generate-content#FinishReason>:
///
/// - `STOP` → `FinishReason::Stop`
/// - `MAX_TOKENS` → `FinishReason::Length`
/// - `SAFETY` / `RECITATION` / `BLOCKLIST` / `PROHIBITED_CONTENT` /
///   `SPII` / `IMAGE_SAFETY` / `LANGUAGE` → `FinishReason::ContentFilter`
/// - `MALFORMED_FUNCTION_CALL` / `UNEXPECTED_TOOL_CALL` / `OTHER` /
///   `FINISH_REASON_UNSPECIFIED` / unknown → `FinishReason::Stop`
///
/// **Audit LOW-3:** `IMAGE_SAFETY` and `LANGUAGE` previously fell
/// through to `Stop` — misleading for tracing, because a customer
/// would see a successful "stop" when Google in fact filtered the
/// response.
fn map_gemini_finish_reason(reason: Option<&str>) -> FinishReason {
    match reason {
        Some("STOP") => FinishReason::Stop,
        Some("MAX_TOKENS") => FinishReason::Length,
        Some("SAFETY")
        | Some("RECITATION")
        | Some("BLOCKLIST")
        | Some("PROHIBITED_CONTENT")
        | Some("SPII")
        | Some("IMAGE_SAFETY")
        | Some("LANGUAGE") => FinishReason::ContentFilter,
        _ => FinishReason::Stop,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Publisher resolution (preserved from skeleton) ──────────────

    #[test]
    fn publisher_resolves_gemini_prefix() {
        assert_eq!(
            VertexPublisher::from_upstream_id("gemini-1.5-pro"),
            Some(VertexPublisher::Google),
        );
        assert_eq!(
            VertexPublisher::from_upstream_id("gemini-2.0-flash-exp"),
            Some(VertexPublisher::Google),
        );
    }

    #[test]
    fn publisher_resolves_anthropic_prefix() {
        assert_eq!(
            VertexPublisher::from_upstream_id("claude-3-5-sonnet@20241022"),
            Some(VertexPublisher::Anthropic),
        );
        assert_eq!(
            VertexPublisher::from_upstream_id("claude-3-haiku@20240307"),
            Some(VertexPublisher::Anthropic),
        );
    }

    #[test]
    fn publisher_resolves_openai_shim_family_and_rawpredict_publishers() {
        // The OpenAI-shim MaaS family — Llama + DeepSeek / Qwen /
        // gpt-oss / MiniMax / Moonshot / Z.ai — all resolve to the one
        // OpenAiCompat variant (one shim wire). Adding a vendor is a
        // resolver-prefix add, nothing else.
        for id in [
            "meta/llama-3.3-70b-instruct-maas",
            "llama3-405b-instruct-maas",
            "deepseek-ai/deepseek-r1-0528-maas",
            "qwen/qwen3-235b-a22b-instruct-maas",
            "openai/gpt-oss-120b-maas",
            "minimaxai/minimax-m2-maas",
            "moonshotai/kimi-k2-instruct-maas",
            "zai-org/glm-4.6-maas",
        ] {
            assert_eq!(
                VertexPublisher::from_upstream_id(id),
                Some(VertexPublisher::OpenAiCompat),
                "{id} should resolve to the OpenAI-shim family",
            );
        }
        // Mistral + AI21 are NOT part of the shim family — they use
        // `:rawPredict` and resolve to their own (still-deferred) arms.
        assert_eq!(
            VertexPublisher::from_upstream_id("mistral-large-2411"),
            Some(VertexPublisher::Mistral),
        );
        assert_eq!(
            VertexPublisher::from_upstream_id("codestral-2501"),
            Some(VertexPublisher::Mistral),
        );
        assert_eq!(
            VertexPublisher::from_upstream_id("jamba-1.5-large"),
            Some(VertexPublisher::Ai21),
        );
    }

    // ─── resolve_api_base precedence ─────────────────────────────────
    //
    // Fixes api7/ai-gateway#391-companion (ai-gateway#390): pre-fix
    // production builds hardcoded `https://<region>-aiplatform.googleapis.com`
    // and silently ignored `ProviderKey.api_base`. The fix adds a
    // `ctx_api_base: Option<&str>` arg with precedence:
    //   #[cfg(test)] override > ctx_api_base > canonical region URL.

    #[test]
    fn resolve_api_base_honors_ctx_api_base_in_production_path() {
        // Constructed without `with_api_base_override` so the test
        // seam is None — exercises the production code path that
        // pre-fix would have hit the canonical URL.
        let bridge = VertexBridge::new();
        let resolved = bridge
            .resolve_api_base("us-central1", Some("http://mock-vertex:8001"))
            .unwrap();
        assert_eq!(resolved, "http://mock-vertex:8001");
    }

    #[test]
    fn resolve_api_base_trims_trailing_slash_on_ctx_override() {
        let bridge = VertexBridge::new();
        let resolved = bridge
            .resolve_api_base("us-central1", Some("http://x.example.internal/"))
            .unwrap();
        assert_eq!(
            resolved, "http://x.example.internal",
            "trailing slash must be stripped so URL stitching produces a single separator",
        );
    }

    #[test]
    fn resolve_api_base_falls_back_to_canonical_on_empty_ctx_value() {
        let bridge = VertexBridge::new();
        // Empty string and whitespace-only values are treated as "no
        // override" — fall through to the canonical region URL.
        assert_eq!(
            bridge.resolve_api_base("us-central1", Some("")).unwrap(),
            "https://us-central1-aiplatform.googleapis.com",
        );
        assert_eq!(
            bridge.resolve_api_base("us-central1", Some("   ")).unwrap(),
            "https://us-central1-aiplatform.googleapis.com",
        );
        assert_eq!(
            bridge.resolve_api_base("us-central1", None).unwrap(),
            "https://us-central1-aiplatform.googleapis.com",
        );
    }

    #[test]
    fn resolve_api_base_cfg_test_override_takes_precedence_over_ctx() {
        // The test-only `with_api_base_override` seam shadows
        // `ctx_api_base` so existing tests that pin against a
        // wiremock URI keep working unchanged.
        let bridge = VertexBridge::new().with_api_base_override("http://wiremock:9999");
        let resolved = bridge
            .resolve_api_base("us-central1", Some("http://ignored-ctx:8001"))
            .unwrap();
        assert_eq!(resolved, "http://wiremock:9999");
    }

    // ─── SSRF / credential-embed validation of api_base (PR #392 audit M1) ──
    //
    // The operator-supplied api_base is interpolated directly into
    // `format!("{base}/v1/projects/...")`. The bridge rejects classes
    // of input that would either escalate (userinfo `@`) or silently
    // corrupt the URL stitching (`?` / `#`). Mirrors the Azure-OpenAI
    // sibling fix (#391) which rejects the same shapes.

    #[test]
    fn resolve_api_base_rejects_non_http_scheme() {
        let bridge = VertexBridge::new();
        for bad in [
            "file:///etc/passwd",
            "gs://my-bucket/path",
            "ftp://internal.example",
            // bare host without scheme — operator might paste this
            // expecting the bridge to add https://, but it won't.
            "mock-vertex:8001",
            "//mock-vertex:8001",
        ] {
            let err = bridge
                .resolve_api_base("us-central1", Some(bad))
                .err()
                .unwrap_or_else(|| panic!("expected error for api_base={bad:?}"));
            match err {
                BridgeError::InvalidUpstreamConfig(msg) => {
                    assert!(
                        msg.contains("http"),
                        "error message should name the http(s) requirement; got: {msg}"
                    );
                }
                other => panic!(
                    "expected InvalidUpstreamConfig error for api_base={bad:?}, got {other:?}"
                ),
            }
        }
    }

    #[test]
    fn resolve_api_base_rejects_embedded_userinfo() {
        let bridge = VertexBridge::new();
        // userinfo (`user:pass@host`) is rejected because:
        //   1. it would leak credentials via access-log URLs;
        //   2. the bridge writes its own Authorization header from
        //      the SA-minted Bearer; userinfo would either shadow it
        //      or get sent to a non-Google host.
        let err = bridge
            .resolve_api_base("us-central1", Some("https://user:secret@proxy.internal"))
            .err()
            .unwrap();
        match err {
            BridgeError::InvalidUpstreamConfig(msg) => {
                assert!(msg.contains("userinfo") || msg.contains("@"));
                // Defense-in-depth: the error message MUST NOT echo
                // the original userinfo back into log output (re-audit
                // LOW-1). The redactor replaces `user:secret` with
                // `<redacted>` so an operator-supplied credential
                // doesn't propagate into operational telemetry.
                assert!(
                    !msg.contains("user:secret"),
                    "error message leaked operator-supplied userinfo: {msg}"
                );
                assert!(
                    msg.contains("<redacted>"),
                    "error message should redact userinfo: {msg}"
                );
            }
            other => panic!("expected InvalidUpstreamConfig error, got {other:?}"),
        }
    }

    #[test]
    fn redact_userinfo_strips_user_and_password() {
        assert_eq!(
            redact_userinfo("https://user:secret@proxy.internal"),
            "https://<redacted>@proxy.internal",
        );
        assert_eq!(
            redact_userinfo("http://just-user@host:8001/path"),
            "http://<redacted>@host:8001/path",
        );
    }

    #[test]
    fn redact_userinfo_leaves_non_userinfo_at_alone() {
        // `@` past the first path slash is NOT userinfo per RFC 3986.
        assert_eq!(
            redact_userinfo("https://proxy.internal/v1@my-namespace"),
            "https://proxy.internal/v1@my-namespace",
        );
        // No scheme → unchanged.
        assert_eq!(
            redact_userinfo("proxy.internal@path"),
            "proxy.internal@path"
        );
        // No `@` at all → unchanged.
        assert_eq!(
            redact_userinfo("https://proxy.internal/x"),
            "https://proxy.internal/x"
        );
    }

    #[test]
    fn resolve_api_base_rejects_query_string() {
        let bridge = VertexBridge::new();
        // Streaming path appends `?alt=sse`; an operator query would
        // silently merge with that and produce a broken URL.
        let err = bridge
            .resolve_api_base(
                "us-central1",
                Some("https://proxy.internal?api-version=oops"),
            )
            .err()
            .unwrap();
        match err {
            BridgeError::InvalidUpstreamConfig(msg) => {
                assert!(msg.contains("query") || msg.contains('?'));
            }
            other => panic!("expected InvalidUpstreamConfig error, got {other:?}"),
        }
    }

    #[test]
    fn resolve_api_base_rejects_fragment() {
        let bridge = VertexBridge::new();
        let err = bridge
            .resolve_api_base("us-central1", Some("https://proxy.internal#fragment"))
            .err()
            .unwrap();
        match err {
            BridgeError::InvalidUpstreamConfig(msg) => {
                assert!(msg.contains("fragment") || msg.contains('#'));
            }
            other => panic!("expected InvalidUpstreamConfig error, got {other:?}"),
        }
    }

    #[test]
    fn resolve_api_base_rejects_embedded_path() {
        // #435: an api_base with a real path segment would silently redirect
        // every upstream call onto the wrong path — reject it with a clear
        // Config error rather than 404-ing the operator later.
        let bridge = VertexBridge::new();
        let err = bridge
            .resolve_api_base("us-central1", Some("https://proxy.internal/evil"))
            .err()
            .unwrap();
        match err {
            BridgeError::InvalidUpstreamConfig(msg) => {
                assert!(
                    msg.contains("bare origin") && msg.contains("no path"),
                    "expected a bare-origin/no-path rejection; got {msg}"
                );
            }
            other => panic!("expected InvalidUpstreamConfig error, got {other:?}"),
        }
    }

    #[test]
    fn resolve_api_base_allows_bare_origin_with_port() {
        // The path rejection must not false-positive on a `:port` origin —
        // `host:port` has no `/` after the scheme.
        let bridge = VertexBridge::new();
        let resolved = bridge
            .resolve_api_base("us-central1", Some("https://proxy.internal:8443"))
            .unwrap();
        assert_eq!(resolved, "https://proxy.internal:8443");
    }

    #[test]
    fn resolve_api_base_rejects_backslash_path() {
        // #464 audit: the WHATWG URL parser the HTTP client uses normalizes
        // `\` to `/` on http(s) URLs, so `host\evil` injects a path just like
        // `host/evil` — it must be rejected the same way.
        let bridge = VertexBridge::new();
        let err = bridge
            .resolve_api_base("us-central1", Some("https://proxy.internal\\evil"))
            .err()
            .unwrap();
        match err {
            BridgeError::InvalidUpstreamConfig(msg) => {
                assert!(
                    msg.contains("bare origin") && msg.contains("no path"),
                    "expected a bare-origin/no-path rejection; got {msg}"
                );
            }
            other => panic!("expected InvalidUpstreamConfig error, got {other:?}"),
        }
    }

    #[test]
    fn publisher_case_insensitive_on_model_name() {
        assert_eq!(
            VertexPublisher::from_upstream_id("Gemini-1.5-Pro"),
            Some(VertexPublisher::Google),
        );
    }

    #[test]
    fn publisher_unknown_prefix_returns_none() {
        assert_eq!(VertexPublisher::from_upstream_id("gpt-4o"), None);
        assert_eq!(VertexPublisher::from_upstream_id(""), None);
    }

    #[test]
    fn publisher_url_segment_matches_vertex_api_path() {
        assert_eq!(
            VertexPublisher::Google.url_segment(),
            Some("publishers/google"),
        );
        assert_eq!(
            VertexPublisher::Anthropic.url_segment(),
            Some("publishers/anthropic"),
        );
        assert_eq!(
            VertexPublisher::Mistral.url_segment(),
            Some("publishers/mistralai"),
        );
        assert_eq!(VertexPublisher::Ai21.url_segment(), Some("publishers/ai21"));
        // The OpenAI-shim family has no `publishers/<vendor>` segment.
        assert_eq!(VertexPublisher::OpenAiCompat.url_segment(), None);
    }

    #[test]
    fn bridge_name_is_stable() {
        assert_eq!(VertexBridge::new().name(), "vertex");
    }

    // ─── VertexSecret parsing ─────────────────────────────────────────

    #[test]
    fn vertex_secret_parses_full_form() {
        let json = r#"{"access_token":"ya29.test","project":"my-proj","region":"us-central1"}"#;
        let s = VertexSecret::parse(json).unwrap();
        assert_eq!(s.access_token.as_deref(), Some("ya29.test"));
        assert!(s.service_account_json.is_none());
        assert_eq!(s.project, "my-proj");
        assert_eq!(s.region, "us-central1");
    }

    #[test]
    fn vertex_secret_rejects_empty() {
        let err = VertexSecret::parse("").unwrap_err();
        match err {
            BridgeError::InvalidUpstreamCredentials(msg) => {
                assert!(msg.contains("api_key is empty"));
            }
            other => panic!("expected InvalidUpstreamCredentials, got {other:?}"),
        }
    }

    #[test]
    fn vertex_secret_rejects_non_json() {
        let err = VertexSecret::parse("ya29.justatoken").unwrap_err();
        match err {
            BridgeError::InvalidUpstreamCredentials(msg) => {
                assert!(msg.contains("must be valid JSON"));
            }
            other => panic!("expected InvalidUpstreamCredentials, got {other:?}"),
        }
    }

    /// Audit-aware: the error message must NOT echo raw secret bytes
    /// (serde error messages can leak partial content).
    #[test]
    fn vertex_secret_error_does_not_leak_secret_content() {
        let leaky = "X-DISTINCTIVE-LEAK-MARKER-Y";
        let err = VertexSecret::parse(leaky).unwrap_err();
        match err {
            BridgeError::InvalidUpstreamCredentials(msg) => {
                assert!(
                    !msg.contains("DISTINCTIVE") && !msg.contains("LEAK-MARKER"),
                    "must NOT leak raw secret bytes; got {msg}"
                );
            }
            other => panic!("expected InvalidUpstreamCredentials, got {other:?}"),
        }
    }

    #[test]
    fn vertex_secret_accepts_service_account_json_path() {
        let json = serde_json::json!({
            "service_account_json": {
                "type": "service_account",
                "private_key": "-----BEGIN PRIVATE KEY-----\nFAKE_PEM_BUT_VALID_HEADER\n-----END PRIVATE KEY-----",
                "client_email": "tester@my-proj.iam.gserviceaccount.com",
                "token_uri": "https://oauth2.googleapis.com/token",
            },
            "project": "my-proj",
            "region": "us-central1"
        });
        let s = VertexSecret::parse(&json.to_string()).unwrap();
        assert!(s.access_token.is_none());
        let sa = s.service_account_json.as_ref().unwrap();
        assert_eq!(sa.client_email, "tester@my-proj.iam.gserviceaccount.com");
        assert_eq!(sa.token_uri, "https://oauth2.googleapis.com/token");
    }

    #[test]
    fn vertex_secret_rejects_both_credential_modes_set() {
        let json = serde_json::json!({
            "access_token": "ya29.foo",
            "service_account_json": {
                "type": "service_account",
                "private_key": "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----",
                "client_email": "x@y.z",
                "token_uri": "https://oauth2.googleapis.com/token",
            },
            "project": "my-proj",
            "region": "us-central1"
        });
        let err = VertexSecret::parse(&json.to_string()).unwrap_err();
        match err {
            BridgeError::InvalidUpstreamCredentials(msg) => {
                assert!(
                    msg.contains("exactly one of access_token or service_account_json"),
                    "got: {msg}"
                );
                assert!(msg.contains("both were provided"));
            }
            other => panic!("expected InvalidUpstreamCredentials, got {other:?}"),
        }
    }

    #[test]
    fn vertex_secret_rejects_neither_credential_mode_set() {
        let json = r#"{"project":"my-proj","region":"us-central1"}"#;
        let err = VertexSecret::parse(json).unwrap_err();
        match err {
            BridgeError::InvalidUpstreamCredentials(msg) => {
                assert!(
                    msg.contains("either access_token or service_account_json"),
                    "got: {msg}"
                );
                assert!(msg.contains("neither was provided"));
            }
            other => panic!("expected InvalidUpstreamCredentials, got {other:?}"),
        }
    }

    #[test]
    fn vertex_secret_rejects_empty_access_token_string() {
        // Edge case: operator pastes an empty string into access_token
        // rather than omitting the field entirely. The pre-mint path
        // would build an Authorization header of "Bearer " which would
        // 401 upstream — better to fail at parse time.
        let json = r#"{"access_token":"","project":"my-proj","region":"us-central1"}"#;
        let err = VertexSecret::parse(json).unwrap_err();
        match err {
            BridgeError::InvalidUpstreamCredentials(msg) => {
                assert!(msg.contains("access_token is empty"), "got: {msg}");
            }
            other => panic!("expected InvalidUpstreamCredentials, got {other:?}"),
        }
    }

    // ─── URL token validation ──────────────────────────────────────────

    #[test]
    fn validate_url_token_accepts_canonical_ids() {
        validate_url_token("project", "my-proj-prod-123").unwrap();
        validate_url_token("region", "us-central1").unwrap();
        validate_url_token("region", "europe-west4").unwrap();
        validate_url_token("upstream_id", "gemini-1.5-pro").unwrap();
        validate_url_token("upstream_id", "gemini-2.0-flash-exp").unwrap();
        // Vertex Claude model ids carry an `@<version>` suffix
        // (e.g. `claude-3-5-sonnet@20241022`). The `@` is a legitimate
        // path char and MUST pass validation — a future tightening of
        // the reject set that banned `@` would break real Claude-on-
        // Vertex dispatch, so pin it here.
        validate_url_token("upstream_id", "claude-3-5-sonnet@20241022").unwrap();
    }

    #[test]
    fn validate_url_token_rejects_url_injection() {
        // Each of these would allow path/query injection if not blocked.
        assert!(matches!(
            validate_url_token("project", "/etc/passwd"),
            Err(BridgeError::Config(_))
        ));
        assert!(matches!(
            validate_url_token("region", "us-central1?alt=evil"),
            Err(BridgeError::Config(_))
        ));
        assert!(matches!(
            validate_url_token("upstream_id", "gemini-1.5-pro#evil"),
            Err(BridgeError::Config(_))
        ));
        assert!(matches!(
            validate_url_token("upstream_id", "gemini-1.5-pro\nfoo"),
            Err(BridgeError::Config(_))
        ));
        assert!(matches!(
            validate_url_token("upstream_id", "gemini/../admin"),
            Err(BridgeError::Config(_))
        ));
    }

    #[test]
    fn validate_url_token_rejects_empty() {
        assert!(matches!(
            validate_url_token("project", ""),
            Err(BridgeError::Config(_))
        ));
    }

    // ─── Gemini request body translation ───────────────────────────────

    #[test]
    fn build_gemini_request_translates_user_turn() {
        let req = ChatFormat::new("my-gemini", vec![ChatMessage::user("hi")]);
        let body = build_gemini_request(&req);
        assert_eq!(body.contents.len(), 1);
        assert_eq!(body.contents[0].role, "user");
        assert_eq!(body.contents[0].parts[0].text, "hi");
        assert!(body.system_instruction.is_none());
        assert!(body.generation_config.is_none());
    }

    #[test]
    fn build_gemini_request_translates_assistant_to_model_role() {
        let req = ChatFormat::new(
            "my-gemini",
            vec![
                ChatMessage::user("hi"),
                ChatMessage::assistant("hello back"),
            ],
        );
        let body = build_gemini_request(&req);
        assert_eq!(body.contents.len(), 2);
        assert_eq!(body.contents[0].role, "user");
        // Gemini uses `model`, NOT `assistant`.
        assert_eq!(body.contents[1].role, "model");
    }

    #[test]
    fn build_gemini_request_lifts_system_to_top_level() {
        let req = ChatFormat::new(
            "my-gemini",
            vec![
                ChatMessage::system("you are helpful"),
                ChatMessage::user("hi"),
            ],
        );
        let body = build_gemini_request(&req);
        // System NOT in contents[].
        assert_eq!(body.contents.len(), 1);
        assert_eq!(body.contents[0].role, "user");
        // System lifted to systemInstruction.
        let sys = body.system_instruction.as_ref().unwrap();
        assert_eq!(sys.parts[0].text, "you are helpful");
    }

    #[test]
    fn build_gemini_request_concatenates_multiple_system_messages() {
        let req = ChatFormat::new(
            "my-gemini",
            vec![
                ChatMessage::system("rule 1"),
                ChatMessage::system("rule 2"),
                ChatMessage::user("hi"),
            ],
        );
        let body = build_gemini_request(&req);
        let sys = body.system_instruction.as_ref().unwrap();
        assert_eq!(sys.parts[0].text, "rule 1\n\nrule 2");
    }

    #[test]
    fn build_gemini_request_emits_generation_config_only_when_set() {
        let mut req = ChatFormat::new("my-gemini", vec![ChatMessage::user("hi")]);
        req.temperature = Some(0.7);
        req.top_p = Some(0.9);
        req.max_tokens = Some(100);
        let body = build_gemini_request(&req);
        let gc = body.generation_config.as_ref().unwrap();
        assert_eq!(gc.temperature, Some(0.7));
        assert_eq!(gc.top_p, Some(0.9));
        assert_eq!(gc.max_output_tokens, Some(100));
    }

    // ─── Gemini response translation ───────────────────────────────────

    #[test]
    fn gemini_response_translates_text_into_chat_response() {
        let raw: GeminiGenerateContentResponse = serde_json::from_str(
            r#"{
                "candidates": [{
                    "content": {"role": "model", "parts": [{"text": "hello"}]},
                    "finishReason": "STOP"
                }],
                "usageMetadata": {
                    "promptTokenCount": 5,
                    "candidatesTokenCount": 1,
                    "totalTokenCount": 6
                }
            }"#,
        )
        .unwrap();
        let chat = gemini_response_into_chat_response(raw, "gemini-1.5-pro");
        assert_eq!(chat.message.content_str(), "hello");
        assert_eq!(chat.message.role, Role::Assistant);
        assert_eq!(chat.finish_reason, FinishReason::Stop);
        assert_eq!(chat.usage.total_tokens, 6);
    }

    #[test]
    fn gemini_response_maps_max_tokens_finish_reason() {
        let raw: GeminiGenerateContentResponse = serde_json::from_str(
            r#"{"candidates": [{"content": {"parts": [{"text": "truncated"}]}, "finishReason": "MAX_TOKENS"}]}"#,
        )
        .unwrap();
        let chat = gemini_response_into_chat_response(raw, "gemini-1.5-pro");
        assert_eq!(chat.finish_reason, FinishReason::Length);
    }

    #[test]
    fn gemini_response_maps_safety_finish_reasons_to_content_filter() {
        // Audit LOW-3: IMAGE_SAFETY and LANGUAGE are content-filter
        // semantics. Mapping them to Stop would mislead tracing — a
        // customer's dashboard would show a successful "stop" when
        // Google in fact filtered the response.
        for r in &[
            "SAFETY",
            "RECITATION",
            "BLOCKLIST",
            "PROHIBITED_CONTENT",
            "SPII",
            "IMAGE_SAFETY",
            "LANGUAGE",
        ] {
            let body = format!(
                r#"{{"candidates": [{{"content": {{"parts": [{{"text": ""}}]}}, "finishReason": {r:?}}}]}}"#
            );
            let raw: GeminiGenerateContentResponse = serde_json::from_str(&body).unwrap();
            let chat = gemini_response_into_chat_response(raw, "gemini-1.5-pro");
            assert_eq!(
                chat.finish_reason,
                FinishReason::ContentFilter,
                "finishReason {r:?} must map to ContentFilter"
            );
        }
    }

    #[test]
    fn gemini_response_handles_missing_usage_metadata() {
        let raw: GeminiGenerateContentResponse = serde_json::from_str(
            r#"{"candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}]}"#,
        )
        .unwrap();
        let chat = gemini_response_into_chat_response(raw, "gemini-1.5-pro");
        assert_eq!(chat.usage.total_tokens, 0);
    }

    // ─── Pre-dispatch validation ───────────────────────────────────────

    use aisix_core::{Model, ProviderKey};
    use std::sync::Arc;

    fn sample_model_with(model_name: &str) -> Arc<Model> {
        let cfg = format!(
            r#"{{
                "display_name": "customer-facing-name",
                "provider": "google",
                "model_name": {model_name:?},
                "provider_key_id": "11111111-1111-1111-1111-111111111111"
            }}"#
        );
        Arc::new(serde_json::from_str(&cfg).unwrap())
    }

    fn sample_pk_with_secret(secret_json: &str) -> Arc<ProviderKey> {
        Arc::new(
            serde_json::from_str(&format!(
                r#"{{"display_name": "vertex-prod", "secret": {}}}"#,
                serde_json::to_string(secret_json).unwrap()
            ))
            .unwrap(),
        )
    }

    /// Variant that sets `api_base` on the ProviderKey — exercises
    /// the production-path override introduced by #390 without
    /// touching the test-only `with_api_base_override` seam.
    fn sample_pk_with_secret_and_api_base(secret_json: &str, api_base: &str) -> Arc<ProviderKey> {
        Arc::new(
            serde_json::from_str(&format!(
                r#"{{"display_name": "vertex-prod", "secret": {}, "api_base": {}}}"#,
                serde_json::to_string(secret_json).unwrap(),
                serde_json::to_string(api_base).unwrap(),
            ))
            .unwrap(),
        )
    }

    fn valid_secret_json() -> &'static str {
        r#"{"access_token":"ya29.test","project":"my-proj","region":"us-central1"}"#
    }

    #[tokio::test]
    async fn chat_with_unknown_publisher_errors_before_dispatch() {
        let bridge = VertexBridge::new();
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("totally-bogus"),
            sample_pk_with_secret(valid_secret_json()),
        );
        let req = ChatFormat::new("customer-facing", vec![ChatMessage::user("hi")]);
        let err = bridge.chat(&req, &ctx).await.unwrap_err();
        match err {
            BridgeError::Config(msg) => {
                assert!(msg.contains("vertex publisher unknown"));
                assert!(msg.contains("totally-bogus"));
            }
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn chat_with_invalid_secret_errors_before_dispatch() {
        let bridge = VertexBridge::new();
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("gemini-1.5-pro"),
            sample_pk_with_secret("not-valid-json"),
        );
        let req = ChatFormat::new("customer-facing", vec![ChatMessage::user("hi")]);
        let err = bridge.chat(&req, &ctx).await.unwrap_err();
        match err {
            BridgeError::InvalidUpstreamCredentials(msg) => {
                assert!(msg.contains("must be valid JSON"));
            }
            other => panic!("expected InvalidUpstreamCredentials, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn chat_with_missing_model_name_errors_before_dispatch() {
        let bridge = VertexBridge::new();
        let model_no_name: Arc<Model> = Arc::new(
            serde_json::from_str(
                r#"{
                    "display_name": "no-upstream-id",
                    "provider": "google",
                    "provider_key_id": "11111111-1111-1111-1111-111111111111"
                }"#,
            )
            .unwrap(),
        );
        let ctx = BridgeContext::new(
            "req-1",
            model_no_name,
            sample_pk_with_secret(valid_secret_json()),
        );
        let req = ChatFormat::new("customer-facing", vec![ChatMessage::user("hi")]);
        let err = bridge.chat(&req, &ctx).await.unwrap_err();
        match err {
            BridgeError::InvalidUpstreamConfig(msg) => {
                assert!(msg.contains("model_name missing"));
            }
            other => panic!("expected InvalidUpstreamConfig error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn chat_ignores_req_model_and_uses_ctx_model_name() {
        let bridge = VertexBridge::new();
        // All real publishers are wired now, so prove "model_name is the
        // source of truth" via the publisher-UNKNOWN path: ctx carries an
        // unrecognized upstream id, req.model carries a *different* bogus
        // value. The resolver error must name the CTX id (not req.model),
        // proving the bridge read model_name from ctx, not the request.
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("unknown-ctx-publisher-xyz"),
            sample_pk_with_secret(valid_secret_json()),
        );
        let req = ChatFormat::new("different-bogus-req-model", vec![ChatMessage::user("hi")]);
        let err = bridge.chat(&req, &ctx).await.unwrap_err();
        match err {
            BridgeError::Config(msg) => {
                assert!(
                    msg.contains("publisher unknown"),
                    "must hit the publisher-unknown path; got {msg}"
                );
                assert!(
                    msg.contains("unknown-ctx-publisher-xyz"),
                    "error must name the CTX model id (proving model_name was used); got {msg}"
                );
                assert!(
                    !msg.contains("different-bogus-req-model"),
                    "error must NOT name req.model; got {msg}"
                );
            }
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    // ─── Dispatch end-to-end against wiremock via api_base override ──

    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, Request as MockRequest, Respond, ResponseTemplate};

    #[derive(Clone, Default)]
    struct CapturingResponder {
        captured_body: std::sync::Arc<std::sync::Mutex<Option<serde_json::Value>>>,
        captured_headers: std::sync::Arc<std::sync::Mutex<Option<http::HeaderMap>>>,
    }

    impl Respond for CapturingResponder {
        fn respond(&self, req: &MockRequest) -> ResponseTemplate {
            let body: serde_json::Value = serde_json::from_slice(&req.body).unwrap_or_default();
            *self.captured_body.lock().unwrap() = Some(body);
            *self.captured_headers.lock().unwrap() = Some(req.headers.clone());
            default_gemini_response_template()
        }
    }

    fn default_gemini_response_template() -> ResponseTemplate {
        ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "candidates": [{
                "content": {"role": "model", "parts": [{"text": "hello from gemini"}]},
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 2,
                "candidatesTokenCount": 4,
                "totalTokenCount": 6
            }
        }))
    }

    #[tokio::test]
    async fn chat_gemini_dispatches_to_generate_content_url() {
        let server = MockServer::start().await;
        let responder = CapturingResponder::default();
        Mock::given(method("POST"))
            .and(path(
                "/v1/projects/my-proj/locations/us-central1/publishers/google/models/gemini-1.5-pro:generateContent",
            ))
            .and(header("authorization", "Bearer ya29.test"))
            .and(header("content-type", "application/json"))
            .respond_with(responder.clone())
            .expect(1)
            .mount(&server)
            .await;

        let bridge = VertexBridge::new().with_api_base_override(server.uri());
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("gemini-1.5-pro"),
            sample_pk_with_secret(valid_secret_json()),
        );
        let req = ChatFormat::new("my-gemini", vec![ChatMessage::user("hi")]);
        let chat = bridge.chat(&req, &ctx).await.unwrap();
        assert_eq!(chat.message.content_str(), "hello from gemini");
        assert_eq!(chat.usage.total_tokens, 6);
    }

    /// Capturing responder that replies with an Anthropic Messages
    /// envelope (Claude on Vertex `:rawPredict` returns the native
    /// Anthropic shape, not the Gemini shape).
    #[derive(Clone, Default)]
    struct CapturingAnthropicResponder {
        captured_body: std::sync::Arc<std::sync::Mutex<Option<serde_json::Value>>>,
    }

    impl Respond for CapturingAnthropicResponder {
        fn respond(&self, req: &MockRequest) -> ResponseTemplate {
            let body: serde_json::Value = serde_json::from_slice(&req.body).unwrap_or_default();
            *self.captured_body.lock().unwrap() = Some(body);
            ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "msg_vertex_test",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-5",
                "content": [{"type": "text", "text": "hello from claude on vertex"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 3, "output_tokens": 5}
            }))
        }
    }

    #[tokio::test]
    async fn chat_anthropic_dispatches_to_raw_predict_url_with_messages_body() {
        let server = MockServer::start().await;
        let responder = CapturingAnthropicResponder::default();
        Mock::given(method("POST"))
            .and(path(
                "/v1/projects/my-proj/locations/us-central1/publishers/anthropic/models/claude-sonnet-4-5:rawPredict",
            ))
            .and(header("authorization", "Bearer ya29.test"))
            .and(header("content-type", "application/json"))
            .respond_with(responder.clone())
            .expect(1)
            .mount(&server)
            .await;

        let bridge = VertexBridge::new().with_api_base_override(server.uri());
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("claude-sonnet-4-5"),
            sample_pk_with_secret(valid_secret_json()),
        );
        let req = ChatFormat::new("my-claude", vec![ChatMessage::user("hi")]);
        let chat = bridge.chat(&req, &ctx).await.unwrap();

        // Customer-visible response decoded from the Anthropic envelope.
        assert_eq!(chat.message.content_str(), "hello from claude on vertex");
        assert_eq!(chat.usage.total_tokens, 8);

        // Wire-shape: Anthropic Messages body with the Vertex
        // anthropic_version, NO `model` (URL carries it), NO `stream`
        // (rawPredict is the non-stream route), user turn present, and
        // `max_tokens` (required by the Anthropic wire, supplied by the
        // shared serializer).
        let body = responder
            .captured_body
            .lock()
            .unwrap()
            .clone()
            .expect("request body captured");
        let obj = body.as_object().expect("object body");
        assert_eq!(
            obj.get("anthropic_version").and_then(|v| v.as_str()),
            Some("vertex-2023-10-16"),
            "Vertex Claude requires anthropic_version=vertex-2023-10-16; got {body}",
        );
        assert!(
            !obj.contains_key("model"),
            "model must be stripped (Vertex keys it off the URL): {body}"
        );
        assert!(
            !obj.contains_key("stream"),
            "stream must be stripped on the :rawPredict (non-stream) route: {body}"
        );
        assert!(
            obj.get("messages")
                .and_then(|v| v.as_array())
                .is_some_and(|m| !m.is_empty()),
            "messages array must carry the user turn: {body}"
        );
        assert!(
            obj.contains_key("max_tokens"),
            "max_tokens is required by the Anthropic Messages wire: {body}"
        );
    }

    /// Capturing responder that returns a native Anthropic SSE stream
    /// (the `:streamRawPredict` wire) while recording the inbound request
    /// body — the streaming analogue of [`CapturingAnthropicResponder`].
    #[derive(Clone, Default)]
    struct CapturingAnthropicStreamResponder {
        captured_body: std::sync::Arc<std::sync::Mutex<Option<serde_json::Value>>>,
    }

    impl Respond for CapturingAnthropicStreamResponder {
        fn respond(&self, req: &MockRequest) -> ResponseTemplate {
            let body: serde_json::Value = serde_json::from_slice(&req.body).unwrap_or_default();
            *self.captured_body.lock().unwrap() = Some(body);
            // Native Anthropic Messages SSE: `message_start` carries the
            // id + model, two `text_delta` content deltas, then
            // `message_delta` (stop_reason + output usage) and the
            // terminal `message_stop`. `event:` lines are included to
            // mirror the real wire — the shared SseDecoder keys off the
            // `data:` payloads and skips the `event:` lines.
            let model = "claude-sonnet-4-5";
            let start = serde_json::json!({
                "type": "message_start",
                "message": {
                    "id": "msg_vertex_stream", "model": model, "role": "assistant",
                    "content": [], "usage": {"input_tokens": 3, "output_tokens": 0}
                }
            });
            let d1 = serde_json::json!({
                "type": "content_block_delta", "index": 0,
                "delta": {"type": "text_delta", "text": "hello"}
            });
            let d2 = serde_json::json!({
                "type": "content_block_delta", "index": 0,
                "delta": {"type": "text_delta", "text": " world"}
            });
            let mdelta = serde_json::json!({
                "type": "message_delta", "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 2}
            });
            let sse = format!(
                "event: message_start\ndata: {}\n\n\
                 event: content_block_delta\ndata: {}\n\n\
                 event: content_block_delta\ndata: {}\n\n\
                 event: message_delta\ndata: {}\n\n\
                 event: message_stop\ndata: {{\"type\":\"message_stop\"}}\n\n",
                serde_json::to_string(&start).unwrap(),
                serde_json::to_string(&d1).unwrap(),
                serde_json::to_string(&d2).unwrap(),
                serde_json::to_string(&mdelta).unwrap(),
            );
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse)
        }
    }

    /// D5.3 — Claude on Vertex streams via the `:streamRawPredict`
    /// action. Sibling of the non-stream `:rawPredict` test above. Pin
    /// the streaming URL action + Bearer auth (mock matchers), the body
    /// shaping that DIFFERS from the non-stream path (`stream:true` is
    /// KEPT, not stripped), the `model` strip + `anthropic_version` add +
    /// `split_system` lift, and that native Anthropic SSE decodes into
    /// ChatChunks whose aggregated content matches the upstream stream
    /// and surface a terminal finish_reason.
    #[tokio::test]
    async fn chat_anthropic_stream_dispatches_to_stream_raw_predict_keeping_stream_in_body() {
        let server = MockServer::start().await;
        let responder = CapturingAnthropicStreamResponder::default();
        Mock::given(method("POST"))
            .and(path(
                "/v1/projects/my-proj/locations/us-central1/publishers/anthropic/models/claude-sonnet-4-5:streamRawPredict",
            ))
            .and(header("authorization", "Bearer ya29.test"))
            .and(header("content-type", "application/json"))
            .respond_with(responder.clone())
            .expect(1)
            .mount(&server)
            .await;

        let bridge = VertexBridge::new().with_api_base_override(server.uri());
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("claude-sonnet-4-5"),
            sample_pk_with_secret(valid_secret_json()),
        );
        let req = ChatFormat::new(
            "my-claude",
            vec![ChatMessage::system("be brief"), ChatMessage::user("hi")],
        );
        let mut stream = bridge.chat_stream(&req, &ctx).await.unwrap();

        let mut content = String::new();
        let mut saw_finish = false;
        while let Some(item) = stream.next().await {
            let chunk = item.unwrap();
            if let Some(delta) = chunk.delta.content.as_deref() {
                content.push_str(delta);
            }
            if chunk.finish_reason.is_some() {
                saw_finish = true;
            }
        }
        assert_eq!(
            content, "hello world",
            "aggregated stream content decodes from the native Anthropic SSE frames"
        );
        assert!(saw_finish, "stream surfaces a terminal finish_reason");

        // Wire-shape: the `:streamRawPredict` body KEEPS `stream:true`
        // (unlike `:rawPredict`, which strips it — see sibling test),
        // strips `model` into the URL, adds the Vertex `anthropic_version`,
        // and lifts the system turn to the top-level `system` field.
        let body = responder
            .captured_body
            .lock()
            .unwrap()
            .clone()
            .expect("request body captured");
        let obj = body.as_object().expect("object body");
        assert_eq!(
            obj.get("stream").and_then(|v| v.as_bool()),
            Some(true),
            "stream:true must be KEPT in the body for :streamRawPredict: {body}"
        );
        assert!(
            !obj.contains_key("model"),
            "model must be stripped (Vertex keys it off the URL): {body}"
        );
        assert_eq!(
            obj.get("anthropic_version").and_then(|v| v.as_str()),
            Some("vertex-2023-10-16"),
            "bridge must add the Vertex anthropic_version: {body}"
        );
        assert!(
            obj.get("system").is_some(),
            "system prompt must be lifted to the top-level `system` field: {body}"
        );
    }

    #[derive(Clone, Default)]
    struct CapturingOpenAiResponder {
        captured_body: std::sync::Arc<std::sync::Mutex<Option<serde_json::Value>>>,
    }

    impl Respond for CapturingOpenAiResponder {
        fn respond(&self, req: &MockRequest) -> ResponseTemplate {
            let body: serde_json::Value = serde_json::from_slice(&req.body).unwrap_or_default();
            *self.captured_body.lock().unwrap() = Some(body);
            // Standard OpenAI chat.completion envelope (the openapi shim
            // returns the OpenAI wire verbatim).
            ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "chatcmpl-vertex-shim",
                "object": "chat.completion",
                "model": "meta/llama-3.3-70b-instruct-maas",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "hello from llama on vertex"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 3, "completion_tokens": 6, "total_tokens": 9}
            }))
        }
    }

    /// Llama-on-Vertex (and the rest of the OpenAI-shim MaaS family)
    /// dispatch through the OpenAI chat-completions shim, NOT a
    /// `publishers/<vendor>/...:rawPredict` URL. Pin the URL shape, the
    /// Bearer auth, and that the model id rides in the body (kept, not
    /// stripped — opposite of the Gemini/Anthropic publisher paths).
    #[tokio::test]
    async fn chat_openai_shim_dispatches_to_openapi_endpoint_with_model_in_body() {
        let server = MockServer::start().await;
        let responder = CapturingOpenAiResponder::default();
        Mock::given(method("POST"))
            .and(path(
                "/v1/projects/my-proj/locations/us-central1/endpoints/openapi/chat/completions",
            ))
            .and(header("authorization", "Bearer ya29.test"))
            .and(header("content-type", "application/json"))
            .respond_with(responder.clone())
            .expect(1)
            .mount(&server)
            .await;

        let bridge = VertexBridge::new().with_api_base_override(server.uri());
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("meta/llama-3.3-70b-instruct-maas"),
            sample_pk_with_secret(valid_secret_json()),
        );
        let req = ChatFormat::new("my-llama", vec![ChatMessage::user("hi")]);
        let chat = bridge.chat(&req, &ctx).await.unwrap();

        // Customer-visible response decoded from the OpenAI envelope.
        assert_eq!(chat.message.content_str(), "hello from llama on vertex");
        assert_eq!(chat.usage.total_tokens, 9);

        // Wire-shape: OpenAI chat body with the model id IN the body
        // (the shim keys off it; there is no model segment in the URL),
        // and the user turn present.
        let body = responder
            .captured_body
            .lock()
            .unwrap()
            .clone()
            .expect("request body captured");
        let obj = body.as_object().expect("object body");
        assert_eq!(
            obj.get("model").and_then(|v| v.as_str()),
            Some("meta/llama-3.3-70b-instruct-maas"),
            "openapi shim keys off the body `model` field (kept, not stripped): {body}"
        );
        assert!(
            obj.get("messages")
                .and_then(|v| v.as_array())
                .is_some_and(|m| !m.is_empty()),
            "messages array must carry the user turn: {body}"
        );
    }

    /// Capturing responder that returns an OpenAI-shim SSE stream while
    /// recording the inbound request body — the streaming-path analogue
    /// of [`CapturingOpenAiResponder`].
    #[derive(Clone, Default)]
    struct CapturingOpenAiStreamResponder {
        captured_body: std::sync::Arc<std::sync::Mutex<Option<serde_json::Value>>>,
    }

    impl Respond for CapturingOpenAiStreamResponder {
        fn respond(&self, req: &MockRequest) -> ResponseTemplate {
            let body: serde_json::Value = serde_json::from_slice(&req.body).unwrap_or_default();
            *self.captured_body.lock().unwrap() = Some(body);
            // OpenAI chat.completion.chunk frames terminated by the
            // `data: [DONE]` sentinel — the exact wire the Vertex openapi
            // shim emits (OpenAI-compatible verbatim): a leading role
            // delta, two content deltas, then a terminal chunk carrying
            // `finish_reason` + `usage`.
            let model = "meta/llama-3.3-70b-instruct-maas";
            let role = serde_json::json!({
                "id": "chatcmpl-vertex-shim", "object": "chat.completion.chunk", "model": model,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]
            });
            let c1 = serde_json::json!({
                "id": "chatcmpl-vertex-shim", "object": "chat.completion.chunk", "model": model,
                "choices": [{"index": 0, "delta": {"content": "hello"}, "finish_reason": null}]
            });
            let c2 = serde_json::json!({
                "id": "chatcmpl-vertex-shim", "object": "chat.completion.chunk", "model": model,
                "choices": [{"index": 0, "delta": {"content": " world"}, "finish_reason": null}]
            });
            let final_chunk = serde_json::json!({
                "id": "chatcmpl-vertex-shim", "object": "chat.completion.chunk", "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
            });
            let sse = format!(
                "data: {}\n\ndata: {}\n\ndata: {}\n\ndata: {}\n\ndata: [DONE]\n\n",
                serde_json::to_string(&role).unwrap(),
                serde_json::to_string(&c1).unwrap(),
                serde_json::to_string(&c2).unwrap(),
                serde_json::to_string(&final_chunk).unwrap(),
            );
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse)
        }
    }

    /// Streaming counterpart to
    /// `chat_openai_shim_dispatches_to_openapi_endpoint_with_model_in_body`.
    /// The OpenAI-shim MaaS family streams through the SAME openapi
    /// endpoint with `stream: true` in the body (the model id still rides
    /// in the body, never the URL). Pin the URL + Bearer auth (via the
    /// mock matchers), the `stream:true` body flag, the model-in-body, and
    /// that the SSE frames decode into ChatChunks whose aggregated content
    /// matches the upstream stream and surface a terminal finish_reason.
    #[tokio::test]
    async fn chat_openai_shim_stream_dispatches_to_openapi_endpoint_with_stream_in_body() {
        let server = MockServer::start().await;
        let responder = CapturingOpenAiStreamResponder::default();
        Mock::given(method("POST"))
            .and(path(
                "/v1/projects/my-proj/locations/us-central1/endpoints/openapi/chat/completions",
            ))
            .and(header("authorization", "Bearer ya29.test"))
            .and(header("content-type", "application/json"))
            .respond_with(responder.clone())
            .expect(1)
            .mount(&server)
            .await;

        let bridge = VertexBridge::new().with_api_base_override(server.uri());
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("meta/llama-3.3-70b-instruct-maas"),
            sample_pk_with_secret(valid_secret_json()),
        );
        let req = ChatFormat::new("my-llama", vec![ChatMessage::user("hi")]);
        let mut stream = bridge.chat_stream(&req, &ctx).await.unwrap();

        let mut content = String::new();
        let mut saw_finish = false;
        while let Some(item) = stream.next().await {
            let chunk = item.unwrap();
            if let Some(delta) = chunk.delta.content.as_deref() {
                content.push_str(delta);
            }
            if chunk.finish_reason.is_some() {
                saw_finish = true;
            }
        }
        assert_eq!(
            content, "hello world",
            "aggregated stream content decodes from the OpenAI SSE frames"
        );
        assert!(saw_finish, "stream surfaces a terminal finish_reason");

        // Wire-shape: `stream:true` and the model id BOTH ride in the
        // body (the shim has no model URL segment); the openapi URL +
        // Bearer are pinned by the mock matchers above.
        let body = responder
            .captured_body
            .lock()
            .unwrap()
            .clone()
            .expect("request body captured");
        let obj = body.as_object().expect("object body");
        assert_eq!(
            obj.get("stream").and_then(|v| v.as_bool()),
            Some(true),
            "stream:true must ride in the body for the openapi shim: {body}"
        );
        assert_eq!(
            obj.get("model").and_then(|v| v.as_str()),
            Some("meta/llama-3.3-70b-instruct-maas"),
            "model id stays in the body (never the URL) on the stream path: {body}"
        );
    }

    /// D5.4 — Mistral on Vertex dispatches through the partner
    /// `publishers/mistralai/models/<model>:rawPredict` URL (NOT the
    /// OpenAI shim, NOT Gemini `:generateContent`) with an OpenAI-shape
    /// body. Reuses the shared OpenAI responder (identical wire). Pin the
    /// URL (publisher segment + model-in-URL + `:rawPredict` action),
    /// Bearer, and that the model is ALSO kept in the body.
    #[tokio::test]
    async fn chat_mistral_dispatches_to_rawpredict_with_model_in_url_and_body() {
        let server = MockServer::start().await;
        let responder = CapturingOpenAiResponder::default();
        Mock::given(method("POST"))
            .and(path(
                "/v1/projects/my-proj/locations/us-central1/publishers/mistralai/models/mistral-large-2411:rawPredict",
            ))
            .and(header("authorization", "Bearer ya29.test"))
            .and(header("content-type", "application/json"))
            .respond_with(responder.clone())
            .expect(1)
            .mount(&server)
            .await;

        let bridge = VertexBridge::new().with_api_base_override(server.uri());
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("mistral-large-2411"),
            sample_pk_with_secret(valid_secret_json()),
        );
        let req = ChatFormat::new("my-mistral", vec![ChatMessage::user("hi")]);
        let chat = bridge.chat(&req, &ctx).await.unwrap();

        // Response decoded from the OpenAI envelope (shared responder).
        assert_eq!(chat.usage.total_tokens, 9);
        assert!(!chat.message.content_str().is_empty());

        // Wire-shape: the model id is kept in the OpenAI body (it rides in
        // BOTH the URL — pinned by the matcher above — and the body).
        let body = responder
            .captured_body
            .lock()
            .unwrap()
            .clone()
            .expect("request body captured");
        let obj = body.as_object().expect("object body");
        assert_eq!(
            obj.get("model").and_then(|v| v.as_str()),
            Some("mistral-large-2411"),
            "Mistral keeps the model id in the OpenAI body (and the URL): {body}"
        );
        assert!(
            obj.get("messages")
                .and_then(|v| v.as_array())
                .is_some_and(|m| !m.is_empty()),
            "messages array must carry the user turn: {body}"
        );
    }

    /// D5.4 — AI21 Jamba on Vertex dispatches through
    /// `publishers/ai21/models/<model>:rawPredict` (a DISTINCT publisher
    /// segment from Mistral) with the same OpenAI-shape body.
    #[tokio::test]
    async fn chat_ai21_dispatches_to_rawpredict_under_ai21_publisher_segment() {
        let server = MockServer::start().await;
        let responder = CapturingOpenAiResponder::default();
        Mock::given(method("POST"))
            .and(path(
                "/v1/projects/my-proj/locations/us-central1/publishers/ai21/models/jamba-1.5-large:rawPredict",
            ))
            .and(header("authorization", "Bearer ya29.test"))
            .respond_with(responder.clone())
            .expect(1)
            .mount(&server)
            .await;

        let bridge = VertexBridge::new().with_api_base_override(server.uri());
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("jamba-1.5-large"),
            sample_pk_with_secret(valid_secret_json()),
        );
        let req = ChatFormat::new("my-jamba", vec![ChatMessage::user("hi")]);
        let chat = bridge.chat(&req, &ctx).await.unwrap();
        assert_eq!(chat.usage.total_tokens, 9);

        let body = responder
            .captured_body
            .lock()
            .unwrap()
            .clone()
            .expect("request body captured");
        assert_eq!(
            body.get("model").and_then(|v| v.as_str()),
            Some("jamba-1.5-large"),
            "AI21 keeps the model id in the OpenAI body (and the URL): {body}"
        );
    }

    /// D5.4 — Mistral / AI21 streaming dispatches to `:streamRawPredict`
    /// with `stream: true` kept in the OpenAI body; the OpenAI SSE
    /// (terminated by `data: [DONE]`) decodes via the shared decoder.
    /// Representative: Mistral (the rail is shared with AI21, differing
    /// only in the publisher URL segment pinned by the non-stream tests).
    #[tokio::test]
    async fn chat_mistral_stream_dispatches_to_streamrawpredict_with_stream_in_body() {
        let server = MockServer::start().await;
        let responder = CapturingOpenAiStreamResponder::default();
        Mock::given(method("POST"))
            .and(path(
                "/v1/projects/my-proj/locations/us-central1/publishers/mistralai/models/mistral-large-2411:streamRawPredict",
            ))
            .and(header("authorization", "Bearer ya29.test"))
            .respond_with(responder.clone())
            .expect(1)
            .mount(&server)
            .await;

        let bridge = VertexBridge::new().with_api_base_override(server.uri());
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("mistral-large-2411"),
            sample_pk_with_secret(valid_secret_json()),
        );
        let req = ChatFormat::new("my-mistral", vec![ChatMessage::user("hi")]);
        let mut stream = bridge.chat_stream(&req, &ctx).await.unwrap();

        let mut content = String::new();
        let mut saw_finish = false;
        while let Some(item) = stream.next().await {
            let chunk = item.unwrap();
            if let Some(delta) = chunk.delta.content.as_deref() {
                content.push_str(delta);
            }
            if chunk.finish_reason.is_some() {
                saw_finish = true;
            }
        }
        assert_eq!(
            content, "hello world",
            "aggregated stream content decodes from the OpenAI SSE frames"
        );
        assert!(saw_finish, "stream surfaces a terminal finish_reason");

        let body = responder
            .captured_body
            .lock()
            .unwrap()
            .clone()
            .expect("request body captured");
        let obj = body.as_object().expect("object body");
        assert_eq!(
            obj.get("stream").and_then(|v| v.as_bool()),
            Some(true),
            "stream:true must ride in the body for :streamRawPredict: {body}"
        );
        assert_eq!(
            obj.get("model").and_then(|v| v.as_str()),
            Some("mistral-large-2411"),
            "model id kept in the body (and the URL) on the stream path: {body}"
        );
    }

    /// D5.4 — AI21 streaming symmetry: same shared rail as Mistral, but
    /// pins the `publishers/ai21` segment on `:streamRawPredict` directly
    /// (not only transitively via the `url_segment()` unit test).
    #[tokio::test]
    async fn chat_ai21_stream_dispatches_to_streamrawpredict_under_ai21_segment() {
        let server = MockServer::start().await;
        let responder = CapturingOpenAiStreamResponder::default();
        Mock::given(method("POST"))
            .and(path(
                "/v1/projects/my-proj/locations/us-central1/publishers/ai21/models/jamba-1.5-large:streamRawPredict",
            ))
            .and(header("authorization", "Bearer ya29.test"))
            .respond_with(responder.clone())
            .expect(1)
            .mount(&server)
            .await;

        let bridge = VertexBridge::new().with_api_base_override(server.uri());
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("jamba-1.5-large"),
            sample_pk_with_secret(valid_secret_json()),
        );
        let req = ChatFormat::new("my-jamba", vec![ChatMessage::user("hi")]);
        let mut stream = bridge.chat_stream(&req, &ctx).await.unwrap();
        while stream.next().await.is_some() {}

        let body = responder
            .captured_body
            .lock()
            .unwrap()
            .clone()
            .expect("request body captured");
        let obj = body.as_object().expect("object body");
        assert_eq!(
            obj.get("stream").and_then(|v| v.as_bool()),
            Some(true),
            "stream:true must ride in the body for AI21 :streamRawPredict: {body}"
        );
        assert_eq!(
            obj.get("model").and_then(|v| v.as_str()),
            Some("jamba-1.5-large"),
            "AI21 keeps the model id in the body (and the URL) on the stream path: {body}"
        );
    }

    /// D5.4 — the partner rail is the one place the model id is BOTH
    /// URL-interpolated and body-serialized, so a hostile `model_name`
    /// must be rejected by the URL-token validator BEFORE any request is
    /// sent. `mistral-large/...` still resolves to the Mistral publisher
    /// (prefix match), then the `/` is rejected — proving no path/query
    /// injection reaches the URL builder.
    #[tokio::test]
    async fn chat_mistral_rejects_model_with_path_injection() {
        let bridge = VertexBridge::new();
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("mistral-large/../admin"),
            sample_pk_with_secret(valid_secret_json()),
        );
        let req = ChatFormat::new("x", vec![ChatMessage::user("hi")]);
        let err = bridge.chat(&req, &ctx).await.unwrap_err();
        assert!(
            matches!(err, BridgeError::Config(_)),
            "path-injection model id must be rejected before dispatch; got {err:?}"
        );
    }

    /// End-to-end production-path coverage for api7/ai-gateway#390:
    /// drive the bridge with `ProviderKey.api_base` set (and NO
    /// `with_api_base_override`), assert the bridge actually POSTs
    /// to the operator-supplied host. Pre-fix this would have hit
    /// the canonical `<region>-aiplatform.googleapis.com` URL, the
    /// wiremock would observe zero requests, and `.expect(1)` would
    /// surface the bug.
    #[tokio::test]
    async fn chat_gemini_honors_provider_key_api_base_in_production_path() {
        let server = MockServer::start().await;
        let responder = CapturingResponder::default();
        Mock::given(method("POST"))
            .and(path(
                "/v1/projects/my-proj/locations/us-central1/publishers/google/models/gemini-1.5-pro:generateContent",
            ))
            .and(header("authorization", "Bearer ya29.test"))
            .respond_with(responder.clone())
            .expect(1) // pre-fix: 0 (traffic went to canonical Google URL); post-fix: 1
            .mount(&server)
            .await;

        // Production path: NO `with_api_base_override`; the URL must
        // be driven entirely by `ProviderKey.api_base`.
        let bridge = VertexBridge::new();
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("gemini-1.5-pro"),
            sample_pk_with_secret_and_api_base(valid_secret_json(), &server.uri()),
        );
        let req = ChatFormat::new("my-gemini", vec![ChatMessage::user("hi")]);
        let chat = bridge.chat(&req, &ctx).await.unwrap();
        assert_eq!(chat.message.content_str(), "hello from gemini");
    }

    /// Same as above but exercises the trailing-slash trim — a
    /// realistic operator paste (`http://corp-proxy/vertex/`) must
    /// still produce a single-slash URL after concatenation.
    #[tokio::test]
    async fn chat_gemini_trims_trailing_slash_on_provider_key_api_base() {
        let server = MockServer::start().await;
        let responder = CapturingResponder::default();
        Mock::given(method("POST"))
            .and(path(
                "/v1/projects/my-proj/locations/us-central1/publishers/google/models/gemini-1.5-pro:generateContent",
            ))
            .respond_with(responder.clone())
            .expect(1)
            .mount(&server)
            .await;

        let bridge = VertexBridge::new();
        // Note: server.uri() typically has no trailing slash; append one.
        let api_base_with_slash = format!("{}/", server.uri());
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("gemini-1.5-pro"),
            sample_pk_with_secret_and_api_base(valid_secret_json(), &api_base_with_slash),
        );
        let req = ChatFormat::new("my-gemini", vec![ChatMessage::user("hi")]);
        let chat = bridge.chat(&req, &ctx).await.unwrap();
        // Body shape unaffected — the assertion is the wiremock's
        // `expect(1)` on the exact path (no `//` doubling).
        assert_eq!(chat.message.content_str(), "hello from gemini");
    }

    #[tokio::test]
    async fn chat_gemini_body_uses_gemini_wire_shape() {
        let server = MockServer::start().await;
        let responder = CapturingResponder::default();
        Mock::given(method("POST"))
            .respond_with(responder.clone())
            .expect(1)
            .mount(&server)
            .await;

        let bridge = VertexBridge::new().with_api_base_override(server.uri());
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("gemini-1.5-pro"),
            sample_pk_with_secret(valid_secret_json()),
        );
        let mut req = ChatFormat::new("my-gemini", vec![ChatMessage::user("hi")]);
        req.temperature = Some(0.5);
        bridge.chat(&req, &ctx).await.unwrap();

        let body = responder.captured_body.lock().unwrap().clone().unwrap();
        // Gemini wire shape pins:
        //   - top-level `contents` array
        //   - role = "user" not "user_message"
        //   - parts[].text (not `content`)
        //   - generationConfig.temperature (camelCase, not snake_case)
        //   - NO `model` field (Vertex puts model in URL)
        //   - NO `stream` field
        let contents = body.get("contents").and_then(|v| v.as_array()).unwrap();
        assert_eq!(contents.len(), 1);
        assert_eq!(
            contents[0].get("role").and_then(|v| v.as_str()),
            Some("user")
        );
        let parts = contents[0].get("parts").and_then(|v| v.as_array()).unwrap();
        assert_eq!(parts[0].get("text").and_then(|v| v.as_str()), Some("hi"));
        let gc = body.get("generationConfig").unwrap();
        assert_eq!(gc.get("temperature").and_then(|v| v.as_f64()), Some(0.5));
        assert!(body.get("model").is_none(), "no model field; body={body}");
        assert!(body.get("stream").is_none(), "no stream field; body={body}");
    }

    #[tokio::test]
    async fn chat_gemini_lifts_system_to_top_level_system_instruction() {
        let server = MockServer::start().await;
        let responder = CapturingResponder::default();
        Mock::given(method("POST"))
            .respond_with(responder.clone())
            .expect(1)
            .mount(&server)
            .await;

        let bridge = VertexBridge::new().with_api_base_override(server.uri());
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("gemini-1.5-pro"),
            sample_pk_with_secret(valid_secret_json()),
        );
        let req = ChatFormat::new(
            "my-gemini",
            vec![
                ChatMessage::system("you are helpful"),
                ChatMessage::user("hi"),
            ],
        );
        bridge.chat(&req, &ctx).await.unwrap();

        let body = responder.captured_body.lock().unwrap().clone().unwrap();
        // System message MUST go in top-level systemInstruction, not in
        // contents[]. Gemini 400s on `role: "system"` in contents.
        let sys = body.get("systemInstruction").unwrap();
        let text = sys
            .get("parts")
            .and_then(|p| p.as_array())
            .and_then(|p| p.first())
            .and_then(|p| p.get("text"))
            .and_then(|v| v.as_str())
            .unwrap();
        assert_eq!(text, "you are helpful");
        // contents[] should have only the user turn.
        let contents = body.get("contents").and_then(|v| v.as_array()).unwrap();
        assert_eq!(contents.len(), 1);
        assert_eq!(
            contents[0].get("role").and_then(|v| v.as_str()),
            Some("user")
        );
    }

    #[tokio::test]
    async fn chat_gemini_uses_model_role_for_assistant_turns() {
        let server = MockServer::start().await;
        let responder = CapturingResponder::default();
        Mock::given(method("POST"))
            .respond_with(responder.clone())
            .expect(1)
            .mount(&server)
            .await;

        let bridge = VertexBridge::new().with_api_base_override(server.uri());
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("gemini-1.5-pro"),
            sample_pk_with_secret(valid_secret_json()),
        );
        let req = ChatFormat::new(
            "my-gemini",
            vec![
                ChatMessage::user("hi"),
                ChatMessage::assistant("hello back"),
                ChatMessage::user("again"),
            ],
        );
        bridge.chat(&req, &ctx).await.unwrap();

        let body = responder.captured_body.lock().unwrap().clone().unwrap();
        let contents = body.get("contents").and_then(|v| v.as_array()).unwrap();
        assert_eq!(contents.len(), 3);
        assert_eq!(
            contents[0].get("role").and_then(|v| v.as_str()),
            Some("user")
        );
        // Gemini's role for assistant is "model", NOT "assistant".
        // A regression that emitted "assistant" would 400 upstream.
        assert_eq!(
            contents[1].get("role").and_then(|v| v.as_str()),
            Some("model"),
            "assistant turn must use role=model; body={body}"
        );
        assert_eq!(
            contents[2].get("role").and_then(|v| v.as_str()),
            Some("user")
        );
    }

    #[tokio::test]
    async fn chat_gemini_authorization_header_carries_pre_minted_bearer() {
        let server = MockServer::start().await;
        let responder = CapturingResponder::default();
        Mock::given(method("POST"))
            .respond_with(responder.clone())
            .expect(1)
            .mount(&server)
            .await;

        let bridge = VertexBridge::new().with_api_base_override(server.uri());
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("gemini-1.5-pro"),
            sample_pk_with_secret(valid_secret_json()),
        );
        let req = ChatFormat::new("my-gemini", vec![ChatMessage::user("hi")]);
        bridge.chat(&req, &ctx).await.unwrap();

        let headers = responder.captured_headers.lock().unwrap().clone().unwrap();
        let auth = headers
            .get("authorization")
            .and_then(|v: &http::HeaderValue| v.to_str().ok())
            .unwrap_or("");
        assert_eq!(
            auth, "Bearer ya29.test",
            "Authorization must carry the pre-minted access_token verbatim"
        );
    }

    #[tokio::test]
    async fn chat_gemini_maps_4xx_to_canned_message_not_body_echo() {
        // Audit-aware: Vertex 4xx error envelopes leak operator project
        // ids. Must redact to canned phrase.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(403).set_body_json(serde_json::json!({
                "error": {
                    "code": 403,
                    "message": "Permission denied on project my-proj-prod-123 for resource gemini-1.5-pro"
                }
            })))
            .mount(&server)
            .await;

        let bridge = VertexBridge::new().with_api_base_override(server.uri());
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("gemini-1.5-pro"),
            sample_pk_with_secret(valid_secret_json()),
        );
        let req = ChatFormat::new("my-gemini", vec![ChatMessage::user("hi")]);
        let err = bridge.chat(&req, &ctx).await.unwrap_err();
        match err {
            BridgeError::UpstreamStatus {
                status,
                message,
                wire,
                parsed,
                ..
            } => {
                assert_eq!(status, 403);
                assert!(
                    !message.contains("my-proj-prod-123")
                        && !message.contains("Permission denied on project"),
                    "upstream body must not leak project id; got message={message:?}"
                );
                // Audit fix (PR #323 MEDIUM-2): pin the wire tag so a
                // refactor that breaks the cross-wire translation
                // pipeline fails this test loudly. parsed.message
                // must stay None for operator-taxonomy redaction.
                assert_eq!(wire, aisix_gateway::UpstreamWire::Vertex);
                if let Some(view) = parsed {
                    assert!(
                        view.message.is_none(),
                        "vertex must NOT surface upstream message (project ids leak); \
                         got {:?}",
                        view.message
                    );
                }
            }
            other => panic!("expected UpstreamStatus, got {other:?}"),
        }
    }

    /// Copilot review (PR #323): non-JSON body (HTML error page from a
    /// fronting load balancer) must NOT trigger serde parsing — the
    /// bridge applies the same content-type guard as
    /// `capture_upstream_error_http`.
    #[tokio::test]
    async fn chat_gemini_non_json_body_skips_envelope_parse() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(403).set_body_raw(
                b"<html><body>403 Forbidden</body></html>".as_slice(),
                "text/html",
            ))
            .mount(&server)
            .await;

        let bridge = VertexBridge::new().with_api_base_override(server.uri());
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("gemini-1.5-pro"),
            sample_pk_with_secret(valid_secret_json()),
        );
        let req = ChatFormat::new("my-gemini", vec![ChatMessage::user("hi")]);
        let err = bridge.chat(&req, &ctx).await.unwrap_err();
        match err {
            BridgeError::UpstreamStatus { parsed, .. } => {
                assert!(
                    parsed.is_none(),
                    "non-JSON body must skip parser; got {parsed:?}"
                );
            }
            other => panic!("expected UpstreamStatus, got {other:?}"),
        }
    }

    /// Audit fix (PR #323 MEDIUM-2): structured-parse path — upstream
    /// returns a Vertex envelope with `error.status` set; the bridge
    /// must extract it as `parsed.kind` for the cross-wire translation
    /// layer to derive an OpenAI `code`. `parsed.message` stays `None`
    /// (operator project ids leak otherwise).
    #[tokio::test]
    async fn chat_gemini_429_populates_parsed_kind_from_grpc_status() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(429).set_body_json(serde_json::json!({
                "error": {
                    "code": 429,
                    "message": "Quota exceeded for project my-secret-proj",
                    "status": "RESOURCE_EXHAUSTED"
                }
            })))
            .mount(&server)
            .await;

        let bridge = VertexBridge::new().with_api_base_override(server.uri());
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("gemini-1.5-pro"),
            sample_pk_with_secret(valid_secret_json()),
        );
        let req = ChatFormat::new("my-gemini", vec![ChatMessage::user("hi")]);
        let err = bridge.chat(&req, &ctx).await.unwrap_err();
        match err {
            BridgeError::UpstreamStatus {
                status,
                message,
                wire,
                parsed,
                ..
            } => {
                assert_eq!(status, 429);
                assert_eq!(wire, aisix_gateway::UpstreamWire::Vertex);
                assert!(
                    !message.contains("my-secret-proj"),
                    "must not leak project id; got {message:?}"
                );
                let parsed = parsed.expect("status field parsed into view");
                assert_eq!(parsed.kind.as_deref(), Some("RESOURCE_EXHAUSTED"));
                assert!(parsed.message.is_none());
            }
            other => panic!("expected UpstreamStatus, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn chat_gemini_rejects_project_with_path_injection() {
        // A malicious secret that injects `/` into the project field
        // must be rejected before URL stitching — otherwise the
        // attacker could redirect to a different path.
        let server = MockServer::start().await;
        let bridge = VertexBridge::new().with_api_base_override(server.uri());
        let evil_secret =
            r#"{"access_token":"ya29","project":"my-proj/../admin","region":"us-central1"}"#;
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("gemini-1.5-pro"),
            sample_pk_with_secret(evil_secret),
        );
        let req = ChatFormat::new("my-gemini", vec![ChatMessage::user("hi")]);
        let err = bridge.chat(&req, &ctx).await.unwrap_err();
        match err {
            BridgeError::Config(msg) => {
                assert!(
                    msg.contains("URL-control characters"),
                    "must reject path injection in project; got {msg}"
                );
            }
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    /// Audit MEDIUM-1: `BridgeError::Config` must NOT echo any byte
    /// of the underlying `InvalidHeaderValue` Display output, because
    /// the bytes being validated ARE the customer's bearer token. A
    /// future change to the `http` crate's Display impl could surface
    /// the offending byte position, leaking partial secret content.
    #[test]
    fn header_invalid_access_token_error_does_not_leak_bytes() {
        // Newline in the access token would let it inject an extra
        // header — header builder must reject AND must not echo the
        // bad bytes back to the customer.
        let err = build_request_headers("ya29.X-DISTINCTIVE-LEAK-Y\nX-Evil: 1", "req-1", None)
            .unwrap_err();
        match err {
            BridgeError::Config(msg) => {
                assert!(
                    !msg.contains("DISTINCTIVE")
                        && !msg.contains("LEAK")
                        && !msg.contains("X-Evil"),
                    "error must NOT echo any token bytes; got {msg}"
                );
                assert!(
                    msg.contains("invalid header characters"),
                    "must still surface the shape error; got {msg}"
                );
            }
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    #[test]
    fn header_invalid_request_id_error_does_not_leak_bytes() {
        let err = build_request_headers("ya29.legit", "req-X-DISTINCTIVE-RID-LEAK-Y\nfoo", None)
            .unwrap_err();
        match err {
            BridgeError::Config(msg) => {
                assert!(
                    !msg.contains("DISTINCTIVE") && !msg.contains("RID-LEAK"),
                    "must NOT leak request_id bytes; got {msg}"
                );
            }
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    /// Audit LOW-4: system-only messages produce empty `contents[]`
    /// (system lifted to top-level). Gemini's schema requires
    /// `contents` to be non-empty; the bridge must fail fast with a
    /// clear error instead of letting Vertex 400 with a generic
    /// "upstream returned 400" surface.
    #[tokio::test]
    async fn chat_gemini_with_system_only_messages_fails_fast() {
        let server = MockServer::start().await;
        // The mock is set up but should NOT be called — the bridge
        // must reject before dispatch. expect(0) catches a regression
        // where the request leaks through.
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200))
            .expect(0)
            .mount(&server)
            .await;

        let bridge = VertexBridge::new().with_api_base_override(server.uri());
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("gemini-1.5-pro"),
            sample_pk_with_secret(valid_secret_json()),
        );
        let req = ChatFormat::new(
            "my-gemini",
            vec![ChatMessage::system("only system, no user")],
        );
        let err = bridge.chat(&req, &ctx).await.unwrap_err();
        match err {
            BridgeError::Config(msg) => {
                assert!(
                    msg.contains("at least one user") && msg.contains("system-only"),
                    "must explain system-only is not supported; got {msg}"
                );
            }
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn chat_gemini_response_with_max_tokens_finish_reason_maps_to_length() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "candidates": [{
                    "content": {"role": "model", "parts": [{"text": "truncated..."}]},
                    "finishReason": "MAX_TOKENS"
                }],
                "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 100, "totalTokenCount": 101}
            })))
            .mount(&server)
            .await;

        let bridge = VertexBridge::new().with_api_base_override(server.uri());
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("gemini-1.5-pro"),
            sample_pk_with_secret(valid_secret_json()),
        );
        let req = ChatFormat::new("my-gemini", vec![ChatMessage::user("hi")]);
        let chat = bridge.chat(&req, &ctx).await.unwrap();
        assert_eq!(chat.finish_reason, FinishReason::Length);
        assert_eq!(chat.message.content_str(), "truncated...");
    }

    // ─── Streaming (:streamGenerateContent?alt=sse) ─────────────────

    // futures::StreamExt is in scope via `use super::*;` (re-exported
    // from the module-level imports), so .next().await works on the
    // ChatChunkStream below without an explicit local import.

    /// SSE body matching what Vertex emits for a short streamed chat:
    /// a content chunk followed by a terminal chunk carrying
    /// `finishReason` + `usageMetadata`. No `[DONE]` sentinel —
    /// Gemini closes the connection cleanly.
    fn happy_path_sse_body() -> String {
        let chunk_1 = serde_json::json!({
            "candidates": [{
                "content": {"role": "model", "parts": [{"text": "hello"}]},
                "index": 0
            }],
            "modelVersion": "gemini-1.5-pro"
        });
        let chunk_2 = serde_json::json!({
            "candidates": [{
                "content": {"role": "model", "parts": [{"text": " world"}]},
                "index": 0
            }],
            "modelVersion": "gemini-1.5-pro"
        });
        let chunk_final = serde_json::json!({
            "candidates": [{
                "content": {"role": "model", "parts": [{"text": ""}]},
                "finishReason": "STOP",
                "index": 0
            }],
            "usageMetadata": {
                "promptTokenCount": 3,
                "candidatesTokenCount": 2,
                "totalTokenCount": 5
            },
            "modelVersion": "gemini-1.5-pro"
        });
        format!(
            "data: {}\n\ndata: {}\n\ndata: {}\n\n",
            serde_json::to_string(&chunk_1).unwrap(),
            serde_json::to_string(&chunk_2).unwrap(),
            serde_json::to_string(&chunk_final).unwrap()
        )
    }

    #[tokio::test]
    async fn chat_gemini_stream_dispatches_to_stream_generate_content_url() {
        // Pin the URL shape: `:streamGenerateContent` action + `?alt=sse`
        // query. wiremock's `path()` matcher covers only the path
        // component, so we assert the body of the *request* via a
        // capturing responder to also verify the query string round
        // trips. A regression that dropped `?alt=sse` would land here
        // — Vertex would return chunked JSON array instead of SSE,
        // breaking the decoder.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path(
                "/v1/projects/my-proj/locations/us-central1/publishers/google/models/gemini-1.5-pro:streamGenerateContent",
            ))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(happy_path_sse_body()),
            )
            .expect(1)
            .mount(&server)
            .await;

        let bridge = VertexBridge::new().with_api_base_override(server.uri());
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("gemini-1.5-pro"),
            sample_pk_with_secret(valid_secret_json()),
        );
        let req = ChatFormat::new("my-gemini", vec![ChatMessage::user("hi")]);
        let mut stream = bridge.chat_stream(&req, &ctx).await.unwrap();
        // Drain the stream so wiremock's `expect(1)` is satisfied on
        // server drop. Errors are surfaced below.
        let mut chunks = Vec::new();
        while let Some(item) = stream.next().await {
            chunks.push(item.unwrap());
        }
        assert!(!chunks.is_empty(), "expected at least one chunk");
    }

    /// Streaming-path counterpart to
    /// `chat_gemini_honors_provider_key_api_base_in_production_path`
    /// (PR #392 audit HIGH-1). Pre-fix, `chat_gemini_stream` also
    /// hardcoded the canonical Google URL; this test pins that the
    /// production path of the streaming bridge follows `ProviderKey.api_base`
    /// without the test-only `with_api_base_override` seam.
    #[tokio::test]
    async fn chat_gemini_stream_honors_provider_key_api_base_in_production_path() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path(
                "/v1/projects/my-proj/locations/us-central1/publishers/google/models/gemini-1.5-pro:streamGenerateContent",
            ))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(happy_path_sse_body()),
            )
            .expect(1) // pre-fix: 0 (canonical Google host); post-fix: 1.
            .mount(&server)
            .await;

        // Production path: NO `with_api_base_override`; the URL must
        // be driven entirely by `ProviderKey.api_base`.
        let bridge = VertexBridge::new();
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("gemini-1.5-pro"),
            sample_pk_with_secret_and_api_base(valid_secret_json(), &server.uri()),
        );
        let req = ChatFormat::new("my-gemini", vec![ChatMessage::user("hi")]);
        let mut stream = bridge.chat_stream(&req, &ctx).await.unwrap();
        let mut chunks = Vec::new();
        while let Some(item) = stream.next().await {
            chunks.push(item.unwrap());
        }
        assert!(
            !chunks.is_empty(),
            "expected at least one chunk via ctx-driven URL"
        );
    }

    #[tokio::test]
    async fn chat_gemini_stream_first_chunk_is_role_then_content() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(happy_path_sse_body()),
            )
            .mount(&server)
            .await;

        let bridge = VertexBridge::new().with_api_base_override(server.uri());
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("gemini-1.5-pro"),
            sample_pk_with_secret(valid_secret_json()),
        );
        let req = ChatFormat::new("my-gemini", vec![ChatMessage::user("hi")]);
        let mut stream = bridge.chat_stream(&req, &ctx).await.unwrap();
        let mut chunks = Vec::new();
        while let Some(item) = stream.next().await {
            chunks.push(item.unwrap());
        }

        // Expected: 4 chunks total
        //   [0] role=Assistant (from chunk_1 first emission)
        //   [1] content="hello" (from chunk_1)
        //   [2] content=" world" (from chunk_2)
        //   [3] finish_reason=Stop + usage (from chunk_final)
        assert_eq!(chunks.len(), 4, "got chunks: {:#?}", chunks);

        assert_eq!(chunks[0].delta.role, Some(Role::Assistant));
        assert!(chunks[0].delta.content.is_none());
        assert!(chunks[0].finish_reason.is_none());
        assert!(chunks[0].usage.is_none());

        assert!(chunks[1].delta.role.is_none());
        assert_eq!(chunks[1].delta.content.as_deref(), Some("hello"));

        assert!(chunks[2].delta.role.is_none());
        assert_eq!(chunks[2].delta.content.as_deref(), Some(" world"));

        assert!(chunks[3].delta.content.is_none());
        assert_eq!(chunks[3].finish_reason, Some(FinishReason::Stop));
        let usage = chunks[3].usage.as_ref().unwrap();
        assert_eq!(usage.prompt_tokens, 3);
        assert_eq!(usage.completion_tokens, 2);
        assert_eq!(usage.total_tokens, 5);
    }

    #[tokio::test]
    async fn chat_gemini_stream_handles_chunks_split_across_packet_boundaries() {
        // SSE decoder must tolerate `data: ` events that straddle
        // HTTP packet boundaries — easy to get wrong with naive
        // chunk-by-chunk parsing. wiremock emits the body in a single
        // write, so this test exercises the SseDecoder's buffering
        // path indirectly: we feed an SSE body that has a `\n\n`
        // separator mid-stream and assert all chunks arrive intact.
        let body = happy_path_sse_body();
        // Sanity: body has multiple chunks separated by \n\n.
        assert!(body.matches("\n\n").count() >= 3);

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(body),
            )
            .mount(&server)
            .await;

        let bridge = VertexBridge::new().with_api_base_override(server.uri());
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("gemini-1.5-pro"),
            sample_pk_with_secret(valid_secret_json()),
        );
        let req = ChatFormat::new("my-gemini", vec![ChatMessage::user("hi")]);
        let mut stream = bridge.chat_stream(&req, &ctx).await.unwrap();
        let mut texts = Vec::new();
        while let Some(item) = stream.next().await {
            let chunk = item.unwrap();
            if let Some(content) = chunk.delta.content {
                texts.push(content);
            }
        }
        // Concatenated text equals the deterministic happy-path body.
        assert_eq!(texts.concat(), "hello world");
    }

    #[tokio::test]
    async fn chat_gemini_stream_finish_reason_maps_max_tokens_to_length() {
        // Sanity-check that the FinishReason mapping applies on the
        // stream path the same way it does on the non-stream path —
        // a regression that hard-coded Stop in the helper would slip
        // past the happy-path test (which uses STOP).
        let body = format!(
            "data: {}\n\n",
            serde_json::to_string(&serde_json::json!({
                "candidates": [{
                    "content": {"role": "model", "parts": [{"text": "truncated"}]},
                    "finishReason": "MAX_TOKENS",
                    "index": 0
                }],
                "usageMetadata": {
                    "promptTokenCount": 1,
                    "candidatesTokenCount": 100,
                    "totalTokenCount": 101
                }
            }))
            .unwrap()
        );

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(body),
            )
            .mount(&server)
            .await;

        let bridge = VertexBridge::new().with_api_base_override(server.uri());
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("gemini-1.5-pro"),
            sample_pk_with_secret(valid_secret_json()),
        );
        let req = ChatFormat::new("my-gemini", vec![ChatMessage::user("hi")]);
        let mut stream = bridge.chat_stream(&req, &ctx).await.unwrap();
        let mut last_finish = None;
        while let Some(item) = stream.next().await {
            let chunk = item.unwrap();
            if chunk.finish_reason.is_some() {
                last_finish = chunk.finish_reason;
            }
        }
        assert_eq!(last_finish, Some(FinishReason::Length));
    }

    #[tokio::test]
    async fn chat_gemini_stream_4xx_returns_upstream_status_error_before_streaming() {
        // 4xx errors must surface BEFORE the stream starts (Bridge
        // returns Err from chat_stream rather than yielding an Err
        // chunk). This matches the non-stream path and is what
        // DP-side error handling expects.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(429).set_body_json(serde_json::json!({
                "error": {
                    "code": 429,
                    "message": "Quota exceeded",
                    "status": "RESOURCE_EXHAUSTED"
                }
            })))
            .mount(&server)
            .await;

        let bridge = VertexBridge::new().with_api_base_override(server.uri());
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("gemini-1.5-pro"),
            sample_pk_with_secret(valid_secret_json()),
        );
        let req = ChatFormat::new("my-gemini", vec![ChatMessage::user("hi")]);
        let err = bridge.chat_stream(&req, &ctx).await.err().unwrap();
        match err {
            BridgeError::UpstreamStatus { status, .. } => {
                assert_eq!(status, 429);
            }
            other => panic!("expected UpstreamStatus, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn chat_gemini_stream_invalid_chunk_payload_surfaces_decode_error() {
        // A regression that mis-parsed the SSE wire (e.g. tried to
        // decode the `data:` payload as a different shape) should
        // surface as UpstreamDecode, not panic or skip the chunk.
        let body = "data: this is not valid JSON\n\n".to_string();
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(body),
            )
            .mount(&server)
            .await;

        let bridge = VertexBridge::new().with_api_base_override(server.uri());
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("gemini-1.5-pro"),
            sample_pk_with_secret(valid_secret_json()),
        );
        let req = ChatFormat::new("my-gemini", vec![ChatMessage::user("hi")]);
        let mut stream = bridge.chat_stream(&req, &ctx).await.unwrap();
        let mut saw_decode_err = false;
        while let Some(item) = stream.next().await {
            if let Err(BridgeError::UpstreamDecode(msg)) = item {
                assert!(msg.contains("vertex stream chunk parse"));
                saw_decode_err = true;
                break;
            }
        }
        assert!(
            saw_decode_err,
            "expected UpstreamDecode error on invalid JSON chunk"
        );
    }

    #[tokio::test]
    async fn chat_gemini_stream_url_uses_alt_sse_query_param() {
        // Pin the URL shape via capturing-responder pattern: assert
        // the request URI included `?alt=sse`. Without that query,
        // Vertex emits a chunked JSON array, not SSE — which would
        // make every downstream stream consumer fail to decode.
        let server = MockServer::start().await;
        let captured: std::sync::Arc<std::sync::Mutex<Option<String>>> =
            std::sync::Arc::new(std::sync::Mutex::new(None));
        let captured_for_responder = captured.clone();
        Mock::given(method("POST"))
            .respond_with(move |req: &MockRequest| {
                *captured_for_responder.lock().unwrap() = Some(req.url.to_string());
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(happy_path_sse_body())
            })
            .mount(&server)
            .await;

        let bridge = VertexBridge::new().with_api_base_override(server.uri());
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("gemini-1.5-pro"),
            sample_pk_with_secret(valid_secret_json()),
        );
        let req = ChatFormat::new("my-gemini", vec![ChatMessage::user("hi")]);
        let mut stream = bridge.chat_stream(&req, &ctx).await.unwrap();
        while stream.next().await.is_some() {}

        let url = captured.lock().unwrap().clone().expect("URL was captured");
        assert!(
            url.contains(":streamGenerateContent"),
            "expected :streamGenerateContent in URL, got {url}"
        );
        assert!(
            url.contains("alt=sse"),
            "expected ?alt=sse query, got {url}"
        );
    }

    // ─── Service-account credential path (D5.1) ─────────────────────

    /// Embedded test SA private key (PEM). Generated via:
    ///   `openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:2048`
    /// Deterministic so the JWT byte sequence is reproducible. NOT a
    /// real GCP key — safe to commit.
    const TEST_SA_PRIVATE_PEM: &str = include_str!("../test-fixtures/test_sa_private.pem");

    fn sa_credential_secret(token_uri: &str) -> String {
        // Operator's secret JSON, SA-mode (no access_token field).
        // Serialize via serde_json so the multi-line PEM lands as a
        // proper JSON-escaped string ("\n" escapes).
        serde_json::json!({
            "service_account_json": {
                "type": "service_account",
                "private_key": TEST_SA_PRIVATE_PEM,
                "client_email": "tester@my-proj.iam.gserviceaccount.com",
                "token_uri": token_uri,
            },
            "project": "my-proj",
            "region": "us-central1"
        })
        .to_string()
    }

    /// End-to-end: SA JSON in secret → bridge resolves token via
    /// in-process minter → minted token used as Authorization Bearer
    /// on the upstream Gemini chat call.
    ///
    /// Pins the full D5.1 pipeline: VertexSecret SA parse +
    /// TokenMinter JWT sign + mint endpoint POST + cache insert +
    /// chat_gemini header forwarding. A regression that broke any
    /// link in this chain surfaces here.
    #[tokio::test]
    async fn chat_gemini_sa_path_mints_token_and_forwards_as_bearer() {
        // Two wiremock servers: one for GCP OAuth token endpoint,
        // one for Vertex AI Gemini chat.
        let oauth_server = MockServer::start().await;
        let vertex_server = MockServer::start().await;

        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "access_token": "ya29.minted-by-mock-oauth",
                "expires_in": 3600,
                "token_type": "Bearer"
            })))
            .expect(1) // exactly one mint despite two chats below
            .mount(&oauth_server)
            .await;

        // Capture the Authorization header on the Vertex chat call so
        // we can assert the bridge actually forwarded the minted
        // token (not a hardcoded placeholder).
        let captured_auth: std::sync::Arc<std::sync::Mutex<Option<String>>> =
            std::sync::Arc::new(std::sync::Mutex::new(None));
        let captured_for_responder = captured_auth.clone();
        Mock::given(method("POST"))
            .and(path(
                "/v1/projects/my-proj/locations/us-central1/publishers/google/models/gemini-1.5-pro:generateContent",
            ))
            .respond_with(move |req: &MockRequest| {
                let auth = req
                    .headers
                    .get("authorization")
                    .and_then(|v| v.to_str().ok())
                    .map(|s| s.to_string());
                *captured_for_responder.lock().unwrap() = auth;
                default_gemini_response_template()
            })
            .mount(&vertex_server)
            .await;

        let bridge = VertexBridge::new()
            .with_api_base_override(vertex_server.uri())
            .with_token_endpoint_override(oauth_server.uri());
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("gemini-1.5-pro"),
            sample_pk_with_secret(&sa_credential_secret(&oauth_server.uri())),
        );
        let req = ChatFormat::new("my-gemini", vec![ChatMessage::user("hi")]);

        // Two chats — second one MUST hit the cache (no second mint).
        let _r1 = bridge.chat(&req, &ctx).await.unwrap();
        let _r2 = bridge.chat(&req, &ctx).await.unwrap();

        let auth = captured_auth
            .lock()
            .unwrap()
            .clone()
            .expect("authorization captured");
        assert_eq!(
            auth, "Bearer ya29.minted-by-mock-oauth",
            "bridge must forward the minted token verbatim"
        );
        // wiremock's .expect(1) on the OAuth mock fires here at drop —
        // proves the cache prevented a second mint on the second chat.
    }

    // ─── RequestOverrides applied on the wire (#339) ───────────────────
    //
    // These pin that the per-`ProviderKey` override pipeline actually
    // reshapes the OUTBOUND Vertex request — not just that the primitives
    // exist (those are unit-tested in `aisix-provider-openai::overrides`).
    // Each test drives a real chat through the bridge against a capturing
    // mock and asserts the recorded body / headers reflect the override.

    /// Build a Vertex `ProviderKey` carrying a `request` override block
    /// (issue #302 §5). The block is supplied as JSON so the test exercises
    /// the same on-disk deserialization cp-api writes to etcd.
    fn sample_pk_with_request_overrides(request: serde_json::Value) -> Arc<ProviderKey> {
        Arc::new(
            serde_json::from_value(serde_json::json!({
                "display_name": "vertex-prod",
                "secret": valid_secret_json(),
                "request": request,
            }))
            .expect("provider_key with request overrides deserializes"),
        )
    }

    #[tokio::test]
    async fn gemini_request_applies_default_body_fields() {
        let server = MockServer::start().await;
        let responder = CapturingResponder::default();
        Mock::given(method("POST"))
            .and(path(
                "/v1/projects/my-proj/locations/us-central1/publishers/google/models/gemini-1.5-pro:generateContent",
            ))
            .respond_with(responder.clone())
            .expect(1)
            .mount(&server)
            .await;

        let bridge = VertexBridge::new().with_api_base_override(server.uri());
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("gemini-1.5-pro"),
            sample_pk_with_request_overrides(serde_json::json!({
                "default_body_fields": { "x_test_default": "injected" }
            })),
        );
        let req = ChatFormat::new("my-gemini", vec![ChatMessage::user("hi")]);
        bridge.chat(&req, &ctx).await.unwrap();

        let body = responder
            .captured_body
            .lock()
            .unwrap()
            .clone()
            .expect("request body captured");
        assert_eq!(
            body.get("x_test_default").and_then(|v| v.as_str()),
            Some("injected"),
            "default_body_fields must inject the absent top-level key into the Gemini body; got {body}",
        );
        // The override pipeline must not disturb the real Gemini payload.
        assert!(
            body.get("contents").and_then(|v| v.as_array()).is_some(),
            "the Gemini `contents` array is preserved alongside the injected field; got {body}",
        );
    }

    #[tokio::test]
    async fn gemini_request_applies_default_headers() {
        let server = MockServer::start().await;
        let responder = CapturingResponder::default();
        Mock::given(method("POST"))
            .and(path(
                "/v1/projects/my-proj/locations/us-central1/publishers/google/models/gemini-1.5-pro:generateContent",
            ))
            .respond_with(responder.clone())
            .expect(1)
            .mount(&server)
            .await;

        let bridge = VertexBridge::new().with_api_base_override(server.uri());
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("gemini-1.5-pro"),
            sample_pk_with_request_overrides(serde_json::json!({
                "default_headers": { "x-vertex-quirk": "on" }
            })),
        );
        let req = ChatFormat::new("my-gemini", vec![ChatMessage::user("hi")]);
        bridge.chat(&req, &ctx).await.unwrap();

        let headers = responder
            .captured_headers
            .lock()
            .unwrap()
            .clone()
            .expect("request headers captured");
        assert_eq!(
            headers.get("x-vertex-quirk").and_then(|v| v.to_str().ok()),
            Some("on"),
            "default_headers must add the absent header to the outbound Vertex request",
        );
    }

    #[tokio::test]
    async fn default_headers_cannot_overwrite_vertex_bearer_auth() {
        // Defense-in-depth: a misconfigured `default_headers` block that
        // names `authorization` must NEVER clobber the minted Vertex OAuth
        // Bearer. A non-reserved companion header still applies, proving the
        // block was processed rather than skipped wholesale.
        let server = MockServer::start().await;
        let responder = CapturingResponder::default();
        Mock::given(method("POST"))
            .and(path(
                "/v1/projects/my-proj/locations/us-central1/publishers/google/models/gemini-1.5-pro:generateContent",
            ))
            .respond_with(responder.clone())
            .expect(1)
            .mount(&server)
            .await;

        let bridge = VertexBridge::new().with_api_base_override(server.uri());
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("gemini-1.5-pro"),
            sample_pk_with_request_overrides(serde_json::json!({
                "default_headers": {
                    "authorization": "Bearer ya29.ATTACKER",
                    "x-ok": "1"
                }
            })),
        );
        let req = ChatFormat::new("my-gemini", vec![ChatMessage::user("hi")]);
        bridge.chat(&req, &ctx).await.unwrap();

        let headers = responder
            .captured_headers
            .lock()
            .unwrap()
            .clone()
            .expect("request headers captured");
        assert_eq!(
            headers.get("authorization").and_then(|v| v.to_str().ok()),
            Some("Bearer ya29.test"),
            "the real minted Bearer must survive a default_headers authorization override",
        );
        assert_eq!(
            headers.get("x-ok").and_then(|v| v.to_str().ok()),
            Some("1"),
            "the non-reserved companion default header still applies",
        );
    }

    #[tokio::test]
    async fn openai_shim_request_applies_param_renames() {
        let server = MockServer::start().await;
        let responder = CapturingOpenAiResponder::default();
        Mock::given(method("POST"))
            .and(path(
                "/v1/projects/my-proj/locations/us-central1/endpoints/openapi/chat/completions",
            ))
            .respond_with(responder.clone())
            .expect(1)
            .mount(&server)
            .await;

        let bridge = VertexBridge::new().with_api_base_override(server.uri());
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("meta/llama-3.3-70b-instruct-maas"),
            sample_pk_with_request_overrides(serde_json::json!({
                "param_renames": { "temperature": "temperature_legacy" }
            })),
        );
        let mut req = ChatFormat::new("my-llama", vec![ChatMessage::user("hi")]);
        req.temperature = Some(0.5);
        bridge.chat(&req, &ctx).await.unwrap();

        let body = responder
            .captured_body
            .lock()
            .unwrap()
            .clone()
            .expect("request body captured");
        assert_eq!(
            body.get("temperature_legacy").and_then(|v| v.as_f64()),
            Some(0.5),
            "param_renames must move `temperature` to the renamed key on the shim body; got {body}",
        );
        assert!(
            body.get("temperature").is_none(),
            "the source key must be gone after the rename; got {body}",
        );
    }

    #[tokio::test]
    async fn openai_shim_request_applies_param_constraints() {
        let server = MockServer::start().await;
        let responder = CapturingOpenAiResponder::default();
        Mock::given(method("POST"))
            .and(path(
                "/v1/projects/my-proj/locations/us-central1/endpoints/openapi/chat/completions",
            ))
            .respond_with(responder.clone())
            .expect(1)
            .mount(&server)
            .await;

        let bridge = VertexBridge::new().with_api_base_override(server.uri());
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("meta/llama-3.3-70b-instruct-maas"),
            sample_pk_with_request_overrides(serde_json::json!({
                "param_constraints": { "temperature_max": 1.0 }
            })),
        );
        let mut req = ChatFormat::new("my-llama", vec![ChatMessage::user("hi")]);
        req.temperature = Some(1.9);
        bridge.chat(&req, &ctx).await.unwrap();

        let body = responder
            .captured_body
            .lock()
            .unwrap()
            .clone()
            .expect("request body captured");
        assert_eq!(
            body.get("temperature").and_then(|v| v.as_f64()),
            Some(1.0),
            "param_constraints must clamp the over-max temperature on the shim body; got {body}",
        );
    }

    #[tokio::test]
    async fn anthropic_request_overrides_run_before_model_strip() {
        // The override pipeline runs BEFORE the Vertex `model` strip, so a
        // `default_body_fields` block can never reintroduce a URL-borne
        // `model` into the `:rawPredict` body — while a genuine extra field
        // still lands.
        let server = MockServer::start().await;
        let responder = CapturingAnthropicResponder::default();
        Mock::given(method("POST"))
            .and(path(
                "/v1/projects/my-proj/locations/us-central1/publishers/anthropic/models/claude-sonnet-4-5:rawPredict",
            ))
            .respond_with(responder.clone())
            .expect(1)
            .mount(&server)
            .await;

        let bridge = VertexBridge::new().with_api_base_override(server.uri());
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("claude-sonnet-4-5"),
            sample_pk_with_request_overrides(serde_json::json!({
                "default_body_fields": { "model": "should-be-stripped", "top_k": 5 }
            })),
        );
        let req = ChatFormat::new("my-claude", vec![ChatMessage::user("hi")]);
        bridge.chat(&req, &ctx).await.unwrap();

        let body = responder
            .captured_body
            .lock()
            .unwrap()
            .clone()
            .expect("request body captured");
        let obj = body.as_object().expect("object body");
        assert!(
            !obj.contains_key("model"),
            "model must stay stripped even when default_body_fields tries to set it; got {body}",
        );
        assert_eq!(
            obj.get("top_k").and_then(|v| v.as_u64()),
            Some(5),
            "a genuine extra default body field still lands on the Anthropic body; got {body}",
        );
        assert_eq!(
            obj.get("anthropic_version").and_then(|v| v.as_str()),
            Some(VERTEX_ANTHROPIC_VERSION),
            "the Vertex anthropic_version shaping still runs after overrides; got {body}",
        );
    }

    /// Build a Vertex `ProviderKey` carrying a `response` override block
    /// (issue #302 §5) — `content_list_to_string` lives here even though it
    /// reshapes the *request* body before send.
    fn sample_pk_with_response_overrides(response: serde_json::Value) -> Arc<ProviderKey> {
        Arc::new(
            serde_json::from_value(serde_json::json!({
                "display_name": "vertex-prod",
                "secret": valid_secret_json(),
                "response": response,
            }))
            .expect("provider_key with response overrides deserializes"),
        )
    }

    #[tokio::test]
    async fn openai_shim_stream_applies_default_body_fields() {
        // Regression guard for the STREAMING body-build path: it serializes
        // the body separately from the non-stream path, so it needs its own
        // proof that the override pipeline reaches the wire.
        let server = MockServer::start().await;
        let responder = CapturingOpenAiStreamResponder::default();
        Mock::given(method("POST"))
            .and(path(
                "/v1/projects/my-proj/locations/us-central1/endpoints/openapi/chat/completions",
            ))
            .respond_with(responder.clone())
            .expect(1)
            .mount(&server)
            .await;

        let bridge = VertexBridge::new().with_api_base_override(server.uri());
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("meta/llama-3.3-70b-instruct-maas"),
            sample_pk_with_request_overrides(serde_json::json!({
                "default_body_fields": { "x_test_default": "injected" }
            })),
        );
        let req = ChatFormat::new("my-llama", vec![ChatMessage::user("hi")]);
        let mut stream = bridge.chat_stream(&req, &ctx).await.unwrap();
        while let Some(item) = stream.next().await {
            let _ = item.unwrap();
        }

        let body = responder
            .captured_body
            .lock()
            .unwrap()
            .clone()
            .expect("request body captured");
        assert_eq!(
            body.get("x_test_default").and_then(|v| v.as_str()),
            Some("injected"),
            "the streaming shim rail must apply default_body_fields too; got {body}",
        );
        assert_eq!(
            body.get("stream").and_then(|v| v.as_bool()),
            Some(true),
            "stream:true stays in the body alongside the override; got {body}",
        );
    }

    #[tokio::test]
    async fn mistral_request_applies_default_body_fields_and_keeps_model_in_body() {
        // Regression guard for the partner `:rawPredict` URL builder rail,
        // which is a distinct code path from the openapi shim.
        let server = MockServer::start().await;
        let responder = CapturingOpenAiResponder::default();
        Mock::given(method("POST"))
            .and(path(
                "/v1/projects/my-proj/locations/us-central1/publishers/mistralai/models/mistral-large-2411:rawPredict",
            ))
            .respond_with(responder.clone())
            .expect(1)
            .mount(&server)
            .await;

        let bridge = VertexBridge::new().with_api_base_override(server.uri());
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("mistral-large-2411"),
            sample_pk_with_request_overrides(serde_json::json!({
                "default_body_fields": { "x_test_default": "injected" }
            })),
        );
        let req = ChatFormat::new("my-mistral", vec![ChatMessage::user("hi")]);
        bridge.chat(&req, &ctx).await.unwrap();

        let body = responder
            .captured_body
            .lock()
            .unwrap()
            .clone()
            .expect("request body captured");
        assert_eq!(
            body.get("x_test_default").and_then(|v| v.as_str()),
            Some("injected"),
            "the partner :rawPredict rail must apply default_body_fields; got {body}",
        );
        // Mistral / AI21 keep the model id in the body (it rides in BOTH the
        // URL and the body) — overrides must not disturb it.
        assert_eq!(
            body.get("model").and_then(|v| v.as_str()),
            Some("mistral-large-2411"),
            "the override pipeline must keep the model the partner rail rides in-body; got {body}",
        );
    }

    #[tokio::test]
    async fn openai_shim_response_content_list_to_string_flattens_outbound_body() {
        // The `content_list_to_string` branch is gated on the *response*
        // override block, so it needs a PK with a `response` section and a
        // multi-block message to observe the array -> string flatten.
        let server = MockServer::start().await;
        let responder = CapturingOpenAiResponder::default();
        Mock::given(method("POST"))
            .and(path(
                "/v1/projects/my-proj/locations/us-central1/endpoints/openapi/chat/completions",
            ))
            .respond_with(responder.clone())
            .expect(1)
            .mount(&server)
            .await;

        let bridge = VertexBridge::new().with_api_base_override(server.uri());
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("meta/llama-3.3-70b-instruct-maas"),
            sample_pk_with_response_overrides(serde_json::json!({
                "content_list_to_string": true
            })),
        );
        // Multi-block text content: ["a", "b", "c"] must flatten to "abc".
        let mut msg = ChatMessage::user("abc");
        msg.content_blocks = Some(vec![
            serde_json::json!({"type": "text", "text": "a"}),
            serde_json::json!({"type": "text", "text": "b"}),
            serde_json::json!({"type": "text", "text": "c"}),
        ]);
        let req = ChatFormat::new("my-llama", vec![msg]);
        bridge.chat(&req, &ctx).await.unwrap();

        let body = responder
            .captured_body
            .lock()
            .unwrap()
            .clone()
            .expect("request body captured");
        assert_eq!(
            body["messages"][0]["content"].as_str(),
            Some("abc"),
            "response.content_list_to_string must flatten array content to a string on the outbound body; got {body}",
        );
    }

    // ─── #723: native embeddings via :predict ─────────────────────────

    #[tokio::test]
    async fn embed_dispatches_predict_and_sums_usage() {
        use wiremock::matchers::{method as wm_method, path as wm_path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(wm_method("POST"))
            .and(wm_path(
                "/v1/projects/my-proj/locations/us-central1/publishers/google/models/text-embedding-005:predict",
            ))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "predictions": [
                    {"embeddings": {"values": [0.1, 0.2], "statistics": {"token_count": 3}}},
                    {"embeddings": {"values": [0.3, 0.4], "statistics": {"token_count": 2}}}
                ]
            })))
            .mount(&server)
            .await;

        let bridge = VertexBridge::new().with_api_base_override(server.uri());
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("text-embedding-005"),
            sample_pk_with_secret(valid_secret_json()),
        );
        let req = EmbeddingRequest {
            model: "customer-facing-name".into(),
            input: vec!["hello".into(), "world".into()],
            input_was_single: false,
            encoding_format: None,
            dimensions: None,
        };
        let resp = bridge.embed(&req, &ctx).await.expect("embed dispatch");
        assert_eq!(resp.object, "list");
        assert_eq!(resp.model, "customer-facing-name");
        assert_eq!(resp.data.len(), 2);
        assert_eq!(resp.data[0].index, 0);
        match &resp.data[0].embedding {
            EmbeddingVector::Float(v) => assert_eq!(v, &vec![0.1, 0.2]),
            other => panic!("expected float vector, got {other:?}"),
        }
        assert_eq!(
            resp.usage.prompt_tokens, 5,
            "token_count sums across predictions"
        );

        // Upstream body: one instance per input, in order, plus the
        // OAuth bearer from the PK secret.
        let received = server.received_requests().await.unwrap();
        assert_eq!(received.len(), 1);
        let body: serde_json::Value = serde_json::from_slice(&received[0].body).unwrap();
        assert_eq!(
            body["instances"],
            serde_json::json!([{"content": "hello"}, {"content": "world"}])
        );
        assert!(
            body.get("parameters").is_none(),
            "no dims -> no parameters key"
        );
        assert_eq!(
            received[0]
                .headers
                .get("authorization")
                .and_then(|v| v.to_str().ok()),
            Some("Bearer ya29.test")
        );
    }

    #[tokio::test]
    async fn embed_forwards_output_dimensionality() {
        use wiremock::matchers::method as wm_method;
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(wm_method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "predictions": [
                    {"embeddings": {"values": [0.5], "statistics": {"token_count": 1}}}
                ]
            })))
            .mount(&server)
            .await;

        let bridge = VertexBridge::new().with_api_base_override(server.uri());
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("gemini-embedding-001"),
            sample_pk_with_secret(valid_secret_json()),
        );
        let req = EmbeddingRequest {
            model: "customer-facing-name".into(),
            input: vec!["hello".into()],
            input_was_single: true,
            encoding_format: None,
            dimensions: Some(256),
        };
        bridge.embed(&req, &ctx).await.expect("embed dispatch");
        let received = server.received_requests().await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&received[0].body).unwrap();
        assert_eq!(body["parameters"]["outputDimensionality"], 256);
    }
}
