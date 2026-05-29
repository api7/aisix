//! `VertexBridge` — family Bridge for [`Adapter::Vertex`].
//!
//! Multi-publisher dispatch for Google Vertex AI. The publisher is
//! resolved from the upstream model id and routed to a per-publisher
//! wire path. **Currently wired:** `google` (Gemini) chat + streaming
//! (`:generateContent` / `:streamGenerateContent`); `anthropic`
//! (Claude) non-stream chat via `:rawPredict` (Anthropic Messages
//! wire). Claude streaming (`:streamRawPredict`, D5.3) and the
//! remaining publishers (`meta` / `mistral` / `ai21`, D5.4) surface
//! clear `not yet implemented` errors — see crate-level docs.
//!
//! Credentials: `ProviderKey.secret` is a JSON-encoded
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
    ChatMessage, ChatResponse, FinishReason, Role, UsageStats,
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
// minus `model`/`stream` plus a Vertex-specific `anthropic_version`.
use aisix_provider_anthropic::wire::{
    build_request as build_anthropic_request, response_into_chat_response, split_system,
    AnthropicResponse,
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
                return Err(BridgeError::Config(format!(
                    "vertex provider_key api_base must use http:// or https:// scheme, got {b:?}",
                )));
            }
            if b.contains('@') {
                // Redact userinfo before echoing the rejected value into
                // the error string — the whole point of rejecting `@` is
                // that operator-pasted credentials shouldn't appear in
                // logs. Audit #392 re-audit LOW-1.
                let redacted = redact_userinfo(b);
                return Err(BridgeError::Config(format!(
                    "vertex provider_key api_base must not embed userinfo (@); use the request's \
                     Authorization header instead, got {redacted:?}",
                )));
            }
            if b.contains('?') {
                return Err(BridgeError::Config(format!(
                    "vertex provider_key api_base must not contain a query string (the bridge \
                     appends `?alt=sse` on streaming; an operator query would silently merge), \
                     got {b:?}",
                )));
            }
            if b.contains('#') {
                return Err(BridgeError::Config(format!(
                    "vertex provider_key api_base must not contain a fragment, got {b:?}",
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
    Client::builder()
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
    /// `publishers/meta/models/llama-*` — Meta's Llama family.
    Meta,
    /// `publishers/mistralai/models/mistral-*` — Mistral on Vertex.
    Mistral,
    /// `publishers/ai21/models/jamba-*` — AI21 Jamba family.
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
        } else if lower.starts_with("meta/") || lower.starts_with("llama") {
            Some(Self::Meta)
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
    /// **Returns `None` for [`Self::Meta`]** — Llama on Vertex uses
    /// the OpenAPI shim at `endpoints/openapi/chat/completions`, not
    /// a `publishers/meta/...` URL.
    pub fn url_segment(self) -> Option<&'static str> {
        Some(match self {
            Self::Google => "publishers/google",
            Self::Anthropic => "publishers/anthropic",
            Self::Mistral => "publishers/mistralai",
            Self::Ai21 => "publishers/ai21",
            Self::Meta => return None,
        })
    }

    /// Human-readable name for the publisher-not-implemented error.
    fn name(&self) -> &'static str {
        match self {
            Self::Google => "google",
            Self::Anthropic => "anthropic",
            Self::Meta => "meta",
            Self::Mistral => "mistralai",
            Self::Ai21 => "ai21",
        }
    }
}

/// `ProviderKey.secret` schema for a Vertex provider key.
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
            return Err(BridgeError::Config(
                "vertex provider_key.secret is empty — \
                 expected JSON with project, region, and either access_token \
                 or service_account_json"
                    .into(),
            ));
        }
        let parsed: VertexSecret = serde_json::from_str(secret).map_err(|_e| {
            BridgeError::Config(
                "vertex provider_key.secret must be valid JSON: \
                 {project, region, and either access_token or service_account_json}"
                    .into(),
            )
        })?;
        // Enforce mutual exclusion. Both-set is suspect (which one
        // wins?); neither-set is unusable. Empty-string token is a
        // distinct error so the operator gets a clearer message than
        // generic "neither set".
        if parsed.access_token.as_deref().is_some_and(str::is_empty) {
            return Err(BridgeError::Config(
                "vertex provider_key.secret.access_token is empty".into(),
            ));
        }
        let has_token = parsed
            .access_token
            .as_deref()
            .is_some_and(|t| !t.is_empty());
        let has_sa = parsed.service_account_json.is_some();
        if has_token && has_sa {
            return Err(BridgeError::Config(
                "vertex provider_key.secret must set exactly one of access_token \
                 or service_account_json (both were provided)"
                    .into(),
            ));
        }
        if !has_token && !has_sa {
            return Err(BridgeError::Config(
                "vertex provider_key.secret must set either access_token or \
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
        .ok_or_else(|| BridgeError::Config("model.model_name missing".into()))
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
            other => Err(BridgeError::Config(format!(
                "vertex publisher {publisher:?} not yet implemented — \
                 tracked under api7/AISIX-Cloud#302 Phase E (D5.4, publisher={})",
                other.name()
            ))),
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
            // Claude on Vertex non-stream chat is wired (`:rawPredict`);
            // streaming (`:streamRawPredict`) is the immediate follow-up
            // — the `stream`-field-in-body vs URL-only contract needs a
            // primary-source confirmation before shipping, so it is held
            // back from this PR rather than guessed.
            VertexPublisher::Anthropic => Err(BridgeError::Config(
                "vertex anthropic (Claude) streaming is not yet implemented — \
                 non-stream chat via :rawPredict is wired; streaming \
                 (:streamRawPredict) tracked under api7/AISIX-Cloud#302 Phase E (D5.3)"
                    .into(),
            )),
            other => Err(BridgeError::Config(format!(
                "vertex publisher {publisher:?} streaming not yet implemented — \
                 tracked under api7/AISIX-Cloud#302 Phase E (D5.3/D5.4, publisher={})",
                other.name()
            ))),
        }
    }
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
        let creds = VertexSecret::parse(&ctx.provider_key.secret)?;
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

        let body = build_gemini_request(req);
        // Audit LOW-4: Gemini requires `contents` to be a non-empty
        // array. If the caller passed system-only messages (lifted to
        // `systemInstruction`), `contents` ends up empty and Vertex
        // returns a generic 400. Fail fast with a clear error so the
        // operator can fix the request shape before the round trip.
        if body.contents.is_empty() {
            return Err(BridgeError::Config(
                "vertex chat: messages must include at least one user / \
                 assistant turn (system-only requests are not supported by Gemini)"
                    .into(),
            ));
        }
        // Resolve bearer: pre-minted token verbatim, or mint+cache
        // via the in-process token minter from SA JSON. Failure
        // surfaces as a Config error (operator-actionable).
        let access_token = creds.resolve_access_token(&self.token_minter).await?;
        let headers = build_request_headers(&access_token, &ctx.request_id)?;
        let client = self.client.clone();
        let started = Instant::now();

        with_deadline(ctx.deadline, started, async move {
            let resp = client
                .post(&url)
                .headers(headers)
                .json(&body)
                .send()
                .await
                .map_err(|e| BridgeError::Transport(e.to_string()))?;

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
        let creds = VertexSecret::parse(&ctx.provider_key.secret)?;
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
            split_system(req).map_err(|e| BridgeError::Config(format!("{e}")))?;
        let anthropic_req = build_anthropic_request(req, upstream_id, system, messages, false);
        let mut body_value = serde_json::to_value(&anthropic_req)
            .map_err(|e| BridgeError::Config(format!("serialize Anthropic request body: {e}")))?;
        if let Some(obj) = body_value.as_object_mut() {
            obj.remove("model");
            obj.remove("stream");
            obj.insert(
                "anthropic_version".to_string(),
                serde_json::Value::String(VERTEX_ANTHROPIC_VERSION.to_string()),
            );
        }

        let access_token = creds.resolve_access_token(&self.token_minter).await?;
        let headers = build_request_headers(&access_token, &ctx.request_id)?;
        let client = self.client.clone();
        let started = Instant::now();

        with_deadline(ctx.deadline, started, async move {
            let resp = client
                .post(&url)
                .headers(headers)
                .json(&body_value)
                .send()
                .await
                .map_err(|e| BridgeError::Transport(e.to_string()))?;

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
        let creds = VertexSecret::parse(&ctx.provider_key.secret)?;
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

        let body = build_gemini_request(req);
        if body.contents.is_empty() {
            return Err(BridgeError::Config(
                "vertex chat: messages must include at least one user / \
                 assistant turn (system-only requests are not supported by Gemini)"
                    .into(),
            ));
        }
        // Resolve bearer (pre-minted OR minted-from-SA) BEFORE
        // entering the stream future so token-mint errors surface
        // as a direct Err return rather than being yielded mid-stream.
        let access_token = creds.resolve_access_token(&self.token_minter).await?;
        let headers = build_request_headers(&access_token, &ctx.request_id)?;
        let client = self.client.clone();
        let started = Instant::now();

        let resp = with_deadline(ctx.deadline, started, async move {
            client
                .post(&url)
                .headers(headers)
                .json(&body)
                .send()
                .await
                .map_err(|e| BridgeError::Transport(e.to_string()))
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
                let bytes: Bytes = item.map_err(|e| BridgeError::Transport(e.to_string()))?;
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
fn build_request_headers(access_token: &str, request_id: &str) -> Result<HeaderMap, BridgeError> {
    if access_token.is_empty() {
        return Err(BridgeError::Config(
            "vertex provider_key.secret.access_token is empty".into(),
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
    Ok(headers)
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
            Role::System => system_parts.push(m.content.clone()),
            Role::User | Role::Tool => contents.push(GeminiContent {
                role: "user",
                parts: vec![GeminiPart {
                    text: m.content.clone(),
                }],
            }),
            Role::Assistant => contents.push(GeminiContent {
                role: "model",
                parts: vec![GeminiPart {
                    text: m.content.clone(),
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
    fn publisher_resolves_meta_mistral_ai21_prefixes() {
        assert_eq!(
            VertexPublisher::from_upstream_id("meta/llama-3.3-70b-instruct-maas"),
            Some(VertexPublisher::Meta),
        );
        assert_eq!(
            VertexPublisher::from_upstream_id("llama3-405b-instruct-maas"),
            Some(VertexPublisher::Meta),
        );
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
                BridgeError::Config(msg) => {
                    assert!(
                        msg.contains("http"),
                        "error message should name the http(s) requirement; got: {msg}"
                    );
                }
                other => panic!("expected Config error for api_base={bad:?}, got {other:?}"),
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
            BridgeError::Config(msg) => {
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
            other => panic!("expected Config error, got {other:?}"),
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
            BridgeError::Config(msg) => {
                assert!(msg.contains("query") || msg.contains('?'));
            }
            other => panic!("expected Config error, got {other:?}"),
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
            BridgeError::Config(msg) => {
                assert!(msg.contains("fragment") || msg.contains('#'));
            }
            other => panic!("expected Config error, got {other:?}"),
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
        assert_eq!(VertexPublisher::Meta.url_segment(), None);
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
            BridgeError::Config(msg) => {
                assert!(msg.contains("secret is empty"));
            }
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    #[test]
    fn vertex_secret_rejects_non_json() {
        let err = VertexSecret::parse("ya29.justatoken").unwrap_err();
        match err {
            BridgeError::Config(msg) => {
                assert!(msg.contains("must be valid JSON"));
            }
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    /// Audit-aware: the error message must NOT echo raw secret bytes
    /// (serde error messages can leak partial content).
    #[test]
    fn vertex_secret_error_does_not_leak_secret_content() {
        let leaky = "X-DISTINCTIVE-LEAK-MARKER-Y";
        let err = VertexSecret::parse(leaky).unwrap_err();
        match err {
            BridgeError::Config(msg) => {
                assert!(
                    !msg.contains("DISTINCTIVE") && !msg.contains("LEAK-MARKER"),
                    "must NOT leak raw secret bytes; got {msg}"
                );
            }
            other => panic!("expected Config error, got {other:?}"),
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
            BridgeError::Config(msg) => {
                assert!(
                    msg.contains("exactly one of access_token or service_account_json"),
                    "got: {msg}"
                );
                assert!(msg.contains("both were provided"));
            }
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    #[test]
    fn vertex_secret_rejects_neither_credential_mode_set() {
        let json = r#"{"project":"my-proj","region":"us-central1"}"#;
        let err = VertexSecret::parse(json).unwrap_err();
        match err {
            BridgeError::Config(msg) => {
                assert!(
                    msg.contains("either access_token or service_account_json"),
                    "got: {msg}"
                );
                assert!(msg.contains("neither was provided"));
            }
            other => panic!("expected Config error, got {other:?}"),
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
            BridgeError::Config(msg) => {
                assert!(msg.contains("access_token is empty"), "got: {msg}");
            }
            other => panic!("expected Config error, got {other:?}"),
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
        assert_eq!(chat.message.content, "hello");
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
    async fn chat_with_non_google_publisher_errors_with_publisher_named() {
        // Claude (anthropic) is now wired for non-stream chat, so use a
        // still-deferred publisher (Mistral) to exercise the
        // publisher-not-implemented path.
        let bridge = VertexBridge::new();
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("mistral-large-2411"),
            sample_pk_with_secret(valid_secret_json()),
        );
        let req = ChatFormat::new("customer-facing", vec![ChatMessage::user("hi")]);
        let err = bridge.chat(&req, &ctx).await.unwrap_err();
        match err {
            BridgeError::Config(msg) => {
                assert!(msg.contains("not yet implemented"));
                assert!(msg.contains("mistral"));
                assert!(msg.contains("D5"));
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
            BridgeError::Config(msg) => {
                assert!(msg.contains("must be valid JSON"));
            }
            other => panic!("expected Config error, got {other:?}"),
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
            BridgeError::Config(msg) => {
                assert!(msg.contains("model_name missing"));
            }
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn chat_ignores_req_model_and_uses_ctx_model_name() {
        let bridge = VertexBridge::new();
        // Use a still-deferred publisher (Mistral) so the proof of
        // "model_name is the source of truth" stays on the
        // publisher-not-implemented path (Claude is now wired).
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("mistral-large-2411"),
            sample_pk_with_secret(valid_secret_json()),
        );
        // req.model set to a value the resolver would reject if it
        // were the source of truth. Bridge must hit publisher-not-
        // implemented (proving it read model_name).
        let req = ChatFormat::new("totally-bogus", vec![ChatMessage::user("hi")]);
        let err = bridge.chat(&req, &ctx).await.unwrap_err();
        match err {
            BridgeError::Config(msg) => {
                assert!(
                    msg.contains("not yet implemented"),
                    "must hit publisher-not-implemented (proving model_name was used); got {msg}"
                );
            }
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn chat_stream_for_non_google_publishers_returns_clear_not_implemented_error() {
        // Streaming is wired for Google (Gemini) only as of D5.2.b.
        // Other publishers (anthropic / meta / mistral / ai21) must
        // still surface the publisher-specific "not yet implemented"
        // error pointing at D5.3 / D5.4 follow-ups so an operator who
        // tries to stream Claude-on-Vertex sees the right tracking
        // reference rather than a generic transport failure.
        // Claude streaming is covered by a dedicated test below (it
        // carries a publisher-specific message now that non-stream
        // Claude is wired). The remaining publishers still surface the
        // generic D5.3/D5.4 streaming-not-implemented error.
        let bridge = VertexBridge::new();
        for upstream in [
            "llama-3-70b-instruct-maas",
            "mistral-large-2411",
            "jamba-1.5-large",
        ] {
            let ctx = BridgeContext::new(
                "req-1",
                sample_model_with(upstream),
                sample_pk_with_secret(valid_secret_json()),
            );
            let req = ChatFormat::new("customer-facing", vec![ChatMessage::user("hi")]);
            let err = bridge.chat_stream(&req, &ctx).await.err().unwrap();
            match err {
                BridgeError::Config(msg) => {
                    assert!(
                        msg.contains("streaming not yet implemented"),
                        "upstream={upstream} unexpected message: {msg}"
                    );
                    assert!(
                        msg.contains("D5.3/D5.4"),
                        "upstream={upstream} missing D5.3/D5.4 reference: {msg}"
                    );
                }
                other => panic!("upstream={upstream} expected Config error, got {other:?}"),
            }
        }
    }

    #[tokio::test]
    async fn chat_stream_anthropic_returns_streaming_not_yet_implemented() {
        // Claude non-stream chat is wired (`:rawPredict`); streaming
        // (`:streamRawPredict`) is the deferred follow-up. The error
        // must be publisher-specific + actionable (point at D5.3 and
        // note that non-stream IS available) rather than a generic
        // transport failure.
        let bridge = VertexBridge::new();
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("claude-3-5-sonnet@20241022"),
            sample_pk_with_secret(valid_secret_json()),
        );
        let req = ChatFormat::new("customer-facing", vec![ChatMessage::user("hi")]);
        let err = bridge.chat_stream(&req, &ctx).await.err().unwrap();
        match err {
            BridgeError::Config(msg) => {
                assert!(
                    msg.contains("streaming is not yet implemented"),
                    "got: {msg}"
                );
                assert!(
                    msg.contains("rawPredict"),
                    "should note non-stream is wired: {msg}"
                );
                assert!(msg.contains("D5.3"), "got: {msg}");
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
        assert_eq!(chat.message.content, "hello from gemini");
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
        assert_eq!(chat.message.content, "hello from claude on vertex");
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
        assert_eq!(chat.message.content, "hello from gemini");
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
        assert_eq!(chat.message.content, "hello from gemini");
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
        let err =
            build_request_headers("ya29.X-DISTINCTIVE-LEAK-Y\nX-Evil: 1", "req-1").unwrap_err();
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
        let err =
            build_request_headers("ya29.legit", "req-X-DISTINCTIVE-RID-LEAK-Y\nfoo").unwrap_err();
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
        assert_eq!(chat.message.content, "truncated...");
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
}
