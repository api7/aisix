//! `VertexBridge` — family Bridge for [`Adapter::Vertex`].
//!
//! Multi-publisher dispatch for Google Vertex AI. The publisher is
//! resolved from the upstream model id and routed to a per-publisher
//! wire path. **Currently wired:** `google` (Gemini) chat. Other
//! publishers + streaming surface clear `not yet implemented` errors
//! referencing D5.x follow-ups — see crate-level docs.
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
    Bridge, BridgeContext, BridgeError, ChatChunkStream, ChatFormat, ChatMessage, ChatResponse,
    FinishReason, Role, UsageStats,
};
use async_trait::async_trait;
use http::{
    header::{HeaderName, HeaderValue},
    HeaderMap,
};
use reqwest::{header, Client, StatusCode};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

use crate::wire;

/// Family Bridge for Google Vertex AI.
pub struct VertexBridge {
    client: Client,
    /// Static `name()` returned to the Hub. Stable across upgrades so
    /// metrics dashboards keep their existing `provider="vertex"`
    /// filters working.
    name: &'static str,
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

    /// Resolve the base host the bridge POSTs to. Production:
    /// `https://<region>-aiplatform.googleapis.com`. Tests can pin
    /// the host via [`Self::with_api_base_override`].
    fn resolve_api_base(&self, region: &str) -> String {
        #[cfg(test)]
        if let Some(b) = &self.api_base_override {
            return b.clone();
        }
        format!("https://{region}-aiplatform.googleapis.com")
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
/// field. `access_token` is a pre-minted OAuth2 bearer (operator
/// manages refresh; ~1-hour GCP TTL). D5.1 follow-up adds in-process
/// JWT signing via `service_account_json`.
#[derive(Debug, Deserialize)]
struct VertexSecret {
    /// Pre-minted GCP OAuth2 access token (operator manages refresh).
    /// D5.1 follow-up will accept a `service_account_json` field in
    /// addition and mint tokens in-process.
    access_token: String,
    /// GCP project id (numeric or named, e.g. `my-org-prod`).
    project: String,
    /// GCP region the Vertex AI deployment targets
    /// (e.g. `us-central1`, `europe-west4`).
    region: String,
}

impl VertexSecret {
    /// Parse the JSON-encoded credential blob.
    ///
    /// **Audit-aware:** error messages MUST NOT echo raw secret
    /// bytes (serde error messages can leak partial content via
    /// "invalid character X at position N").
    fn parse(secret: &str) -> Result<Self, BridgeError> {
        if secret.trim().is_empty() {
            return Err(BridgeError::Config(
                "vertex provider_key.secret is empty — \
                 expected JSON {access_token, project, region}"
                    .into(),
            ));
        }
        serde_json::from_str::<VertexSecret>(secret).map_err(|_e| {
            BridgeError::Config(
                "vertex provider_key.secret must be valid JSON: \
                 {access_token, project, region}"
                    .into(),
            )
        })
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
/// leak operator project ids. We map to canned status-keyed phrases.
async fn map_http_error(status: StatusCode, resp: reqwest::Response) -> BridgeError {
    let retry_after = aisix_gateway::parse_retry_after(resp.headers());
    let _ = resp.text().await; // drain body, discard content
    let message = match status.as_u16() {
        401 | 403 => "upstream authentication failed".to_string(),
        404 => "upstream model or endpoint not found".to_string(),
        408 => "upstream request timeout".to_string(),
        429 => "upstream rate limited".to_string(),
        _ => format!("upstream returned {}", status.as_u16()),
    };
    BridgeError::upstream_status_with_retry_after(status.as_u16(), message, retry_after)
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
            other => Err(BridgeError::Config(format!(
                "vertex publisher {publisher:?} not yet implemented — \
                 tracked under api7/AISIX-Cloud#302 Phase E (D5.3/D5.4, publisher={})",
                other.name()
            ))),
        }
    }

    async fn chat_stream(
        &self,
        _req: &ChatFormat,
        ctx: &BridgeContext,
    ) -> Result<ChatChunkStream, BridgeError> {
        let upstream_id = upstream_model(ctx)?;
        let _publisher = VertexPublisher::from_upstream_id(upstream_id).ok_or_else(|| {
            BridgeError::Config(format!(
                "vertex publisher unknown for upstream model id {upstream_id:?}; \
                 expected one of gemini-* / claude-* / meta/llama-* or llama* / \
                 mistral-* / jamba-*"
            ))
        })?;
        Err(BridgeError::Config(
            "vertex streaming is not yet implemented — \
             tracked under api7/AISIX-Cloud#302 Phase E (D5.2.b)"
                .into(),
        ))
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

        let base = self.resolve_api_base(&creds.region);
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
        let headers = build_request_headers(&creds.access_token, &ctx.request_id)?;
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
        assert_eq!(s.access_token, "ya29.test");
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

    // ─── URL token validation ──────────────────────────────────────────

    #[test]
    fn validate_url_token_accepts_canonical_ids() {
        validate_url_token("project", "my-proj-prod-123").unwrap();
        validate_url_token("region", "us-central1").unwrap();
        validate_url_token("region", "europe-west4").unwrap();
        validate_url_token("upstream_id", "gemini-1.5-pro").unwrap();
        validate_url_token("upstream_id", "gemini-2.0-flash-exp").unwrap();
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
        let bridge = VertexBridge::new();
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("claude-3-5-sonnet@20241022"),
            sample_pk_with_secret(valid_secret_json()),
        );
        let req = ChatFormat::new("customer-facing", vec![ChatMessage::user("hi")]);
        let err = bridge.chat(&req, &ctx).await.unwrap_err();
        match err {
            BridgeError::Config(msg) => {
                assert!(msg.contains("not yet implemented"));
                assert!(msg.contains("anthropic"));
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
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("claude-3-5-sonnet@20241022"),
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
    async fn chat_stream_returns_clear_not_implemented_error() {
        let bridge = VertexBridge::new();
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("gemini-1.5-pro"),
            sample_pk_with_secret(valid_secret_json()),
        );
        let req = ChatFormat::new("customer-facing", vec![ChatMessage::user("hi")]);
        let err = bridge.chat_stream(&req, &ctx).await.err().unwrap();
        match err {
            BridgeError::Config(msg) => {
                assert!(msg.contains("streaming is not yet implemented"));
                assert!(msg.contains("D5.2.b"));
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
                status, message, ..
            } => {
                assert_eq!(status, 403);
                assert!(
                    !message.contains("my-proj-prod-123")
                        && !message.contains("Permission denied on project"),
                    "upstream body must not leak project id; got message={message:?}"
                );
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
}
