//! `AzureOpenAiBridge` — family Bridge for [`Adapter::AzureOpenai`].
//!
//! Wire shape is OpenAI chat-completions (parsers reused from
//! `aisix-provider-openai::wire`). Azure differs on three axes:
//!
//! 1. **URL pattern** — deployment-keyed:
//!    `https://<resource>.openai.azure.com/openai/deployments/<deployment>/chat/completions?api-version=<version>`
//! 2. **Auth header** — `api-key: <secret>` (NOT `Authorization: Bearer`)
//! 3. **Response extension** — Azure adds `prompt_filter_results` /
//!    `content_filter_results` blocks; the reused OpenAI parsers
//!    tolerate them via serde's default-deny-on-known behavior.
//!
//! Override apply pipeline (request body + headers) mirrors
//! `OpenAiBridge` and reuses the helpers from
//! `aisix_provider_openai::overrides`.

use aisix_core::{RequestOverrides, ResponseOverrides, StreamDoneMarker};
use aisix_gateway::{
    Bridge, BridgeContext, BridgeError, ChatChunk, ChatChunkStream, ChatFormat, ChatResponse,
    SseDecoder, SseEvent,
};
use async_trait::async_trait;
use futures::StreamExt;
use http::{
    header::{HeaderName, HeaderValue},
    HeaderMap,
};
use reqwest::{header, Client, StatusCode};
use serde_json::Value;
use std::time::{Duration, Instant};

use aisix_provider_openai::overrides::{
    apply_content_list_to_string, apply_default_body_fields, apply_default_headers,
    apply_param_constraints, apply_param_renames, apply_stream_done_marker_policy,
    extract_reasoning_field, StreamDoneOutcome,
};
use aisix_provider_openai::wire::{
    build_request, messages_from, response_into_chat_response, stream_chunk_into_chat_chunk,
    OpenAiResponse, OpenAiStreamChunk,
};

use crate::aad_token_mint::TokenMinter;
use crate::wire;

use std::sync::Arc;

/// Family Bridge for Azure OpenAI Service.
pub struct AzureOpenAiBridge {
    client: Client,
    /// Static `name()` returned to the Hub. Distinct from `"openai"`
    /// so dashboards can split Azure traffic from canonical OpenAI
    /// traffic in metrics.
    name: &'static str,
    /// In-process AAD token cache + minter. Used only when the
    /// inbound `ProviderKey.secret` parses to the AAD branch; the
    /// api-key branch bypasses this entirely. `Arc` so the bridge
    /// remains cheaply clonable for callers that share it across
    /// Hub registrations.
    token_minter: Arc<TokenMinter>,
    /// Test-only POST URL override. When set, [`Bridge::chat`] /
    /// [`Bridge::chat_stream`] still run resolve / validation /
    /// header / body building against the real `AzureUpstreamRef`,
    /// but the final HTTP request goes to this URL instead of
    /// `<resource>.openai.azure.com`. Lets wiremock cover the
    /// full bridge entry point (not only the helper sub-fns).
    #[cfg(test)]
    url_override: Option<String>,
}

impl AzureOpenAiBridge {
    /// Construct an Azure OpenAI bridge with the canonical name
    /// `"azure-openai"`. The Hub looks this up via [`Bridge::name`]
    /// when emitting per-request metrics (provider label).
    pub fn new() -> Self {
        Self::with_client(default_client())
    }

    /// Construct an Azure OpenAI bridge with a caller-supplied
    /// [`reqwest::Client`]. Useful when downstream callers want to
    /// share a connection pool with other bridges or pin custom
    /// timeouts. Public surface — not test-only.
    pub fn with_client(client: Client) -> Self {
        Self {
            token_minter: Arc::new(TokenMinter::new(client.clone())),
            client,
            name: "azure-openai",
            #[cfg(test)]
            url_override: None,
        }
    }

    /// Test-only seam: replace the AAD token endpoint host on the
    /// internal minter (without touching the chat-completions URL
    /// override on `url_override`). Used by AAD-flow tests so
    /// `client_credentials` POSTs land on a wiremock instance.
    #[cfg(test)]
    pub(crate) fn with_aad_token_endpoint_override(mut self, url: impl Into<String>) -> Self {
        self.token_minter =
            Arc::new(TokenMinter::new(self.client.clone()).with_token_endpoint_override(url));
        self
    }

    /// Resolve the URL the bridge will POST to. Returns
    /// `upstream.chat_completions_url()` in production; tests can
    /// override via [`Self::with_url_override`].
    fn resolve_url(&self, upstream: &AzureUpstreamRef) -> String {
        #[cfg(test)]
        if let Some(u) = &self.url_override {
            return u.clone();
        }
        upstream.chat_completions_url()
    }

    /// Test-only seam: rewrite the POST URL so wiremock can stand
    /// in for `<resource>.openai.azure.com`. Header / body / resolve
    /// / validation paths still run normally against the canonical
    /// api_base configured on the ProviderKey.
    #[cfg(test)]
    pub(crate) fn with_url_override(mut self, url: impl Into<String>) -> Self {
        self.url_override = Some(url.into());
        self
    }

    /// Resolve the per-request auth pair from the provider key's
    /// secret. Returns either an `api_key` (verbatim resource-key)
    /// or a `bearer_token` (freshly minted-or-cached AAD access
    /// token). Token-mint failures (e.g. AAD 401 invalid_client)
    /// surface as `Err` here so `chat()` / `chat_stream()` short-
    /// circuit BEFORE attempting the upstream Azure OpenAI call.
    async fn resolve_auth(&self, ctx: &BridgeContext) -> Result<AzureAuth, BridgeError> {
        match AzureSecret::parse(&ctx.provider_key.secret)? {
            AzureSecret::ApiKey(key) => Ok(AzureAuth {
                api_key: Some(key),
                bearer_token: None,
            }),
            AzureSecret::Aad(creds) => {
                let token = self.token_minter.get_token(&creds).await?;
                Ok(AzureAuth {
                    api_key: None,
                    bearer_token: Some(token),
                })
            }
        }
    }
}

impl Default for AzureOpenAiBridge {
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

/// Parsed Azure upstream reference resolved from a provider_key's
/// `api_base` + the request's upstream model id.
///
/// Azure's chat-completions URL pattern (per
/// <https://learn.microsoft.com/en-us/azure/ai-services/openai/reference>):
///
/// ```text
/// https://<resource>.openai.azure.com/openai/deployments/<deployment>/chat/completions?api-version=<version>
/// ```
///
/// - `resource` — the Azure resource name, e.g. `acme-prod-west-us`
/// - `deployment` — operator-named deployment, e.g. `gpt4o-prod`
/// - `api_version` — Azure's date-stamped API version, e.g. `2024-10-21`
///
/// The resolver is intentionally cautious — any missing piece produces
/// a clear `BridgeError::Config` so an operator can fix the
/// registration before traffic ever hits Azure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AzureUpstreamRef {
    pub resource: String,
    pub deployment: String,
    pub api_version: String,
}

impl AzureUpstreamRef {
    /// Most recent GA REST API version at crate publish time.
    /// Operators **must** pin an explicit version via
    /// `provider_key.api_base` for production traffic — this constant
    /// is a stop-gap default. Azure deprecates older versions on a
    /// published schedule:
    /// <https://learn.microsoft.com/en-us/azure/ai-services/openai/api-version-deprecation>.
    ///
    /// Pinned at a GA shape (`YYYY-MM-DD`, no `-preview` suffix) so a
    /// future bump can't silently re-introduce a preview default.
    pub const DEFAULT_API_VERSION: &'static str = "2024-10-21";

    /// Resolve from the deployment name + an optional pre-parsed
    /// `api_base`.
    ///
    /// Both `deployment` and the resolved `resource` are validated to
    /// match a strict `[A-Za-z0-9_-]+` shape: Azure resource names
    /// and deployment names are constrained to that set per the
    /// portal, and a URL-injection vector via `?`, `#`, `/`, or
    /// whitespace would let an operator-supplied default redirect
    /// the dispatch to an attacker-pinned API version.
    pub fn resolve(deployment: &str, api_base: Option<&str>) -> Result<Self, BridgeError> {
        validate_url_token("deployment name", deployment)?;

        let base = api_base.unwrap_or_default().trim();
        let resource = if base.is_empty() {
            return Err(BridgeError::Config(
                "azure provider_key has no api_base — \
                 expected https://<resource>.openai.azure.com or a bare resource name"
                    .into(),
            ));
        } else if let Some(rest) = base
            .strip_prefix("https://")
            .or_else(|| base.strip_prefix("http://"))
        {
            // Canonical form: split off the leading host segment
            // before the first `.`. The remainder of the host MUST
            // be `openai.azure.com` — anything else is a misconfig
            // we surface up rather than silently dropping the path.
            let (host_resource, host_tail) = rest.split_once('.').ok_or_else(|| {
                BridgeError::Config(format!(
                    "azure api_base {base:?} missing the .openai.azure.com suffix"
                ))
            })?;
            // Strip a trailing `/...` path so an operator who pasted
            // the full chat-completions URL still parses correctly,
            // but reject anything that injected query params.
            let host_tail_trimmed = host_tail.trim_end_matches('/');
            let host_tail_core = host_tail_trimmed
                .split_once('/')
                .map(|(host, _path)| host)
                .unwrap_or(host_tail_trimmed);
            if host_tail_core != "openai.azure.com" {
                return Err(BridgeError::Config(format!(
                    "azure api_base {base:?} host must end in .openai.azure.com \
                     (got host suffix {host_tail_core:?})"
                )));
            }
            host_resource.to_string()
        } else {
            // Bare-resource shorthand.
            base.to_string()
        };

        validate_url_token("resource name", &resource)?;

        Ok(Self {
            resource,
            deployment: deployment.to_string(),
            api_version: Self::DEFAULT_API_VERSION.to_string(),
        })
    }

    /// Build the chat-completions URL for this Azure upstream.
    pub fn chat_completions_url(&self) -> String {
        format!(
            "https://{}.openai.azure.com/openai/deployments/{}/chat/completions?api-version={}",
            self.resource, self.deployment, self.api_version,
        )
    }
}

/// Reject URL-control characters in operator/customer-supplied tokens
/// that end up in the Azure URL path. Azure resource names and
/// deployment names are documented as `[A-Za-z0-9_-]+`, so anything
/// outside that set is either a misconfig or a URL-injection attempt
/// (e.g. `?api-version=evil` to override the bridge's version pin).
fn validate_url_token(name: &str, value: &str) -> Result<(), BridgeError> {
    if value.is_empty() {
        return Err(BridgeError::Config(format!(
            "azure {name} is empty (expected an identifier matching [A-Za-z0-9_-]+)"
        )));
    }
    if !value
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
    {
        return Err(BridgeError::Config(format!(
            "azure {name} {value:?} contains URL-control characters — \
             must match [A-Za-z0-9_-]+ (no spaces, slashes, dots, query params, or hash)"
        )));
    }
    Ok(())
}

/// Discriminated auth scheme. Today's resource-key deployments keep
/// the verbatim-string secret shape (backward-compat); the AAD path
/// is opted into by encoding the secret as JSON
/// `{tenant_id, client_id, client_secret}`.
///
/// Detection is by the leading character of the trimmed secret —
/// `{` triggers JSON parse, anything else is treated as a literal
/// api-key. Real Azure api-keys are URL-safe base64 strings, never
/// starting with `{`, so the heuristic is unambiguous in practice.
#[derive(Debug)]
pub(crate) enum AzureSecret {
    ApiKey(String),
    Aad(crate::aad_token_mint::AadCredentials),
}

impl AzureSecret {
    /// Parse the inbound `ProviderKey.secret` into either the api-key
    /// or AAD branch.
    ///
    /// **Audit-aware:** error messages MUST NOT echo raw secret
    /// bytes. The branch heuristic is `starts_with('{')` after
    /// trimming — that test does not panic, returns Bool, no leaks.
    /// The JSON parse error message is fixed (not interpolated from
    /// serde) so partial secret contents can't surface in the error.
    pub(crate) fn parse(secret: &str) -> Result<Self, BridgeError> {
        let trimmed = secret.trim();
        if trimmed.is_empty() {
            return Err(BridgeError::Config("provider_key.secret is empty".into()));
        }
        if trimmed.starts_with('{') {
            let creds: crate::aad_token_mint::AadCredentials = serde_json::from_str(trimmed)
                .map_err(|_e| {
                    BridgeError::Config(
                        "azure provider_key.secret looks JSON-shaped but failed to parse \
                         as AAD client_credentials \
                         {tenant_id, client_id, client_secret}"
                            .into(),
                    )
                })?;
            creds.validate()?;
            Ok(AzureSecret::Aad(creds))
        } else {
            Ok(AzureSecret::ApiKey(trimmed.to_string()))
        }
    }
}

/// Resolved per-request auth header pair the bridge writes onto the
/// outbound request. Exactly ONE of `api_key` / `bearer_token` is
/// `Some`; the other is `None`.
pub(crate) struct AzureAuth {
    pub api_key: Option<String>,
    pub bearer_token: Option<String>,
}

/// Pull the upstream deployment name off the BridgeContext. Azure
/// deployment names (operator-defined in the Azure portal, e.g.
/// `gpt4o-prod`) live on Model.model_name. `req.model` is the
/// customer-facing display name and must NOT be used here.
fn upstream_model(ctx: &BridgeContext) -> Result<&str, BridgeError> {
    ctx.model
        .model_name
        .as_deref()
        .ok_or_else(|| BridgeError::Config("model.model_name missing".into()))
}

/// Map an Azure HTTP error response to a customer-visible
/// [`BridgeError::UpstreamStatus`].
///
/// **Audit M1 — sensitive-info redaction:** Azure error envelopes
/// (`{"error": {"code": "...", "message": "..."}}`) often include the
/// operator-defined deployment id (e.g. "The API deployment for this
/// resource does not exist.") or the resource hostname. The
/// customer-visible `message` stays a canned phrase. We do read
/// `error.code` into [`UpstreamErrorView::kind`] — these are a small
/// closed set of Azure-defined tokens (`DeploymentNotFound`,
/// `content_filter`, …), stable taxonomy rather than operator data —
/// so the envelope-translation layer can derive an OpenAI `code`.
/// [`UpstreamErrorView::message`] stays `None`.
///
/// Azure-specific quirk: content-policy violations nest under
/// `error.inner_error.code` *or* `error.innererror.code` (Azure emits
/// both casings depending on endpoint).
/// `"ResponsibleAIPolicyViolation"` on the inner code overrides the
/// outer `code` so the gateway can recognise the policy hit even when
/// the outer code is the generic `invalid_request_error`.
async fn map_http_error(status: StatusCode, resp: reqwest::Response) -> BridgeError {
    let retry_after = aisix_gateway::parse_retry_after(resp.headers());
    let is_json = aisix_gateway::response_is_json(&resp);
    let body =
        aisix_gateway::read_body_capped(resp, aisix_gateway::MAX_UPSTREAM_ERROR_BODY_BYTES).await;
    // Skip the serde parse on non-JSON bodies (HTML error page from a
    // load-balancer / front door). Same guard as
    // `capture_upstream_error_http`.
    let kind = is_json.then(|| parse_azure_error_code(&body)).flatten();
    let message = match status.as_u16() {
        401 | 403 => "upstream authentication failed".to_string(),
        404 => "upstream deployment or model not found".to_string(),
        408 => "upstream request timeout".to_string(),
        409 => "upstream conflict".to_string(),
        413 => "upstream request entity too large".to_string(),
        429 => "upstream rate limited".to_string(),
        _ => format!("upstream returned {}", status.as_u16()),
    };
    // Azure's envelope only has a single `error.code` field (no
    // separate `type`). Downstream OpenAI clients expect both
    // `error.type` AND `error.code` — populate both `kind` and `code`
    // from the upstream token so the translation layer can either
    // pass it through (for OpenAI-compat tokens like
    // `rate_limit_exceeded`) or override with a derived code (for
    // Azure-specific tokens like `DeploymentNotFound`).
    let parsed = kind.as_ref().map(|k| {
        Box::new(aisix_gateway::UpstreamErrorView {
            kind: Some(k.clone()),
            message: None,
            code: Some(k.clone()),
            param: None,
        })
    });
    BridgeError::UpstreamStatus {
        status: status.as_u16(),
        message,
        parsed,
        wire: aisix_gateway::UpstreamWire::AzureOpenAI,
        retry_after,
    }
}

/// Extract the Azure error code from the envelope, applying the
/// `inner_error` / `innererror` content-policy quirk.
fn parse_azure_error_code(body: &[u8]) -> Option<String> {
    #[derive(serde::Deserialize)]
    struct Outer {
        error: Inner,
    }
    #[derive(serde::Deserialize)]
    struct Inner {
        code: Option<String>,
        #[serde(rename = "inner_error")]
        inner_error: Option<InnerInner>,
        innererror: Option<InnerInner>,
    }
    #[derive(serde::Deserialize)]
    struct InnerInner {
        code: Option<String>,
    }
    let outer: Outer = serde_json::from_slice(body).ok()?;
    let inner_code = outer
        .error
        .inner_error
        .as_ref()
        .or(outer.error.innererror.as_ref())
        .and_then(|i| i.code.clone());
    // Content-policy violation on `inner_error` overrides the outer
    // generic code.
    if inner_code.as_deref() == Some("ResponsibleAIPolicyViolation") {
        return inner_code;
    }
    outer.error.code
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

/// Apply RequestOverrides + ResponseOverrides flag-driven body
/// transforms before sending. Mirrors `OpenAiBridge::prepare_outbound_body`.
fn prepare_outbound_body<T: serde::Serialize>(
    typed: &T,
    request: Option<&RequestOverrides>,
    response: Option<&ResponseOverrides>,
) -> Result<Value, BridgeError> {
    let mut body = serde_json::to_value(typed)
        .map_err(|e| BridgeError::Config(format!("serialize request body: {e}")))?;
    if let Some(r) = request {
        apply_param_renames(&mut body, &r.param_renames);
        if let Some(constraints) = &r.param_constraints {
            apply_param_constraints(&mut body, constraints);
        }
        apply_default_body_fields(&mut body, &r.default_body_fields);
    }
    if response.is_some_and(|r| r.content_list_to_string) {
        apply_content_list_to_string(&mut body);
    }
    Ok(body)
}

/// Build the base outbound `HeaderMap` for Azure. The auth header
/// depends on the resolved auth scheme:
///
///   - api-key scheme → `api-key: <secret>` (Azure docs:
///     <https://learn.microsoft.com/en-us/azure/ai-services/openai/reference>;
///     the literal lowercase-hyphenated `api-key`, NOT
///     `Authorization: Bearer`).
///   - AAD scheme → `Authorization: Bearer <access_token>` (the
///     industry-standard Bearer header; matches how Azure's own
///     Python SDK sets the header on Entra ID auth).
///
/// In both branches the bridge also sets `Content-Type: application/json`,
/// `x-aisix-request-id: <ctx.request_id>`, and (for streaming)
/// `Accept: text/event-stream`. Bridge-owned headers are inserted
/// before `apply_default_headers` so the reserved-headers list in
/// `aisix-provider-openai::overrides` (which already covers
/// `api-key`, `authorization`, `x-api-key`, plus hop-by-hop /
/// proxy-auth headers) cannot overwrite them. Defense in depth: the
/// reserved-list blocks even before the `headers.contains_key` guard
/// inside `apply_default_headers`.
fn build_request_headers(
    auth: &AzureAuth,
    request_id: &str,
    sse: bool,
    request: Option<&RequestOverrides>,
) -> Result<HeaderMap, BridgeError> {
    let mut headers = HeaderMap::new();
    match (&auth.api_key, &auth.bearer_token) {
        (Some(key), None) => {
            let value = HeaderValue::from_str(key).map_err(|e| {
                BridgeError::Config(format!("api key contains invalid header chars: {e}"))
            })?;
            headers.insert(HeaderName::from_static("api-key"), value);
        }
        (None, Some(token)) => {
            // Bearer token from AAD. Same byte-validation pattern as
            // api-key. Note: the validation error MUST NOT echo the
            // token bytes — `http::InvalidHeaderValue`'s Display is
            // opaque, but a future change could include byte
            // position; rebind to a fixed message to be safe.
            let value = HeaderValue::from_str(&format!("Bearer {token}")).map_err(|_| {
                BridgeError::Config(
                    "azure aad access_token contains invalid header characters".into(),
                )
            })?;
            headers.insert(header::AUTHORIZATION, value);
        }
        // parse() / resolve_auth() ensure exactly one is Some — keep
        // explicit guard for defense in depth.
        _ => {
            return Err(BridgeError::Config(
                "internal: AzureAuth must set exactly one of api_key / bearer_token".into(),
            ))
        }
    }
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    let rid = HeaderValue::from_str(request_id).map_err(|e| {
        BridgeError::Config(format!("request_id contains invalid header chars: {e}"))
    })?;
    headers.insert(HeaderName::from_static("x-aisix-request-id"), rid);
    if sse {
        headers.insert(
            header::ACCEPT,
            HeaderValue::from_static("text/event-stream"),
        );
    }
    if let Some(r) = request {
        apply_default_headers(&mut headers, &r.default_headers);
    }
    Ok(headers)
}

#[async_trait]
impl Bridge for AzureOpenAiBridge {
    fn name(&self) -> &'static str {
        self.name
    }

    async fn chat(
        &self,
        req: &ChatFormat,
        ctx: &BridgeContext,
    ) -> Result<ChatResponse, BridgeError> {
        let deployment = upstream_model(ctx)?;
        let upstream = AzureUpstreamRef::resolve(deployment, ctx.provider_key.api_base.as_deref())?;
        // Keep reserved-config helpers reachable from the public surface
        // so a future override-validation PR can wire them in without
        // re-exposing private state.
        let _ = wire::reserved_query_params();
        let _ = wire::reserved_auth_headers();

        // Resolve auth BEFORE entering the request future so AAD
        // mint failures surface as a direct Err return rather than
        // a transport timeout. api-key path is a no-op string copy.
        let auth = self.resolve_auth(ctx).await?;
        // Azure expects the deployment name in the URL path; the JSON
        // body's `model` field is ignored by Azure (or echoed back).
        // We still set it to the deployment name for log-trace clarity
        // and to mirror the upstream OpenAI SDK convention.
        let messages = messages_from(req);
        let typed = build_request(req, deployment, &messages, false);
        let body = prepare_outbound_body(
            &typed,
            ctx.provider_key.request.as_ref(),
            ctx.provider_key.response.as_ref(),
        )?;
        let headers = build_request_headers(
            &auth,
            &ctx.request_id,
            false,
            ctx.provider_key.request.as_ref(),
        )?;
        let url = self.resolve_url(&upstream);
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

            // Azure injects `prompt_filter_results` /
            // `content_filter_results` blocks. OpenAiResponse uses
            // `#[serde(default)]` on optional fields and does NOT set
            // `deny_unknown_fields`, so the extension fields pass
            // through transparently without breaking deserialization.
            let parsed: OpenAiResponse = resp
                .json()
                .await
                .map_err(|e| BridgeError::UpstreamDecode(e.to_string()))?;
            Ok(response_into_chat_response(parsed))
        })
        .await
    }

    async fn chat_stream(
        &self,
        req: &ChatFormat,
        ctx: &BridgeContext,
    ) -> Result<ChatChunkStream, BridgeError> {
        let deployment = upstream_model(ctx)?;
        let upstream = AzureUpstreamRef::resolve(deployment, ctx.provider_key.api_base.as_deref())?;
        let _ = wire::reserved_query_params();
        let _ = wire::reserved_auth_headers();

        // See chat() — resolve auth before the request future.
        let auth = self.resolve_auth(ctx).await?;
        let messages = messages_from(req);
        let typed = build_request(req, deployment, &messages, true);
        let body = prepare_outbound_body(
            &typed,
            ctx.provider_key.request.as_ref(),
            ctx.provider_key.response.as_ref(),
        )?;
        let headers = build_request_headers(
            &auth,
            &ctx.request_id,
            true,
            ctx.provider_key.request.as_ref(),
        )?;
        let url = self.resolve_url(&upstream);
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

        // Snapshot the response-side override knobs onto the stream
        // closure so it can run after `ctx` drops.
        let reasoning_path = ctx
            .provider_key
            .response
            .as_ref()
            .and_then(|r| r.reasoning_field.clone());
        let done_marker_policy = ctx
            .provider_key
            .response
            .as_ref()
            .and_then(|r| r.stream_done_marker);
        let bridge_name = self.name;
        let request_id_for_log = ctx.request_id.clone();

        // Audit H2: thread the deadline into the streaming loop so a
        // slow / hanging upstream body can't wedge the connection
        // after headers arrive. The post-headers POST already covered
        // the initial wait; this covers the per-chunk wait too.
        let stream_deadline = ctx.deadline.map(|d| started + d);

        let byte_stream = resp.bytes_stream();
        let stream = build_chunk_stream(
            byte_stream,
            reasoning_path,
            done_marker_policy,
            bridge_name,
            request_id_for_log,
            stream_deadline,
            started,
        );
        Ok(Box::pin(stream))
    }
}

fn build_chunk_stream<S>(
    byte_stream: S,
    reasoning_path: Option<String>,
    done_marker_policy: Option<StreamDoneMarker>,
    bridge_name: &'static str,
    request_id: String,
    deadline: Option<Instant>,
    started: Instant,
) -> impl futures::Stream<Item = Result<ChatChunk, BridgeError>> + Send
where
    S: futures::Stream<Item = reqwest::Result<bytes::Bytes>> + Send + 'static,
{
    async_stream::try_stream! {
        let mut decoder = SseDecoder::new();
        let mut stream = Box::pin(byte_stream);
        let mut done_marker_seen = false;
        'outer: loop {
            // Audit H2: enforce deadline on each per-chunk wait. The
            // first POST is already covered upstream; this covers the
            // body-streaming phase. `None` deadline disables timeout.
            let next = match deadline {
                Some(d) => match tokio::time::timeout_at(d.into(), stream.next()).await {
                    Ok(item) => item,
                    Err(_) => {
                        Err(BridgeError::Timeout {
                            elapsed_ms: started.elapsed().as_millis() as u64,
                        })?;
                        unreachable!()
                    }
                },
                None => stream.next().await,
            };
            let Some(next) = next else { break 'outer; };
            let chunk = next.map_err(|e| BridgeError::Transport(e.to_string()))?;
            for event in decoder.feed(chunk.as_ref()) {
                match event {
                    SseEvent::Done => {
                        done_marker_seen = true;
                        break 'outer;
                    }
                    SseEvent::Data(payload) => {
                        let parsed = parse_stream_chunk(&payload, reasoning_path.as_deref())?;
                        yield stream_chunk_into_chat_chunk(parsed);
                    }
                }
            }
        }
        match decoder.finish() {
            Some(SseEvent::Done) => {
                done_marker_seen = true;
            }
            Some(SseEvent::Data(payload)) => {
                let parsed = parse_stream_chunk(&payload, reasoning_path.as_deref())?;
                yield stream_chunk_into_chat_chunk(parsed);
            }
            None => {}
        }
        // Issue #302 §5 `response.stream_done_marker` — violations
        // are logged (operator diagnostic) but never error the
        // request: customer chunks have already been delivered.
        if let Some(policy) = done_marker_policy {
            match apply_stream_done_marker_policy(policy, done_marker_seen) {
                StreamDoneOutcome::Ok => {}
                StreamDoneOutcome::MissingDoneMarker => {
                    tracing::warn!(
                        bridge = bridge_name,
                        request_id = %request_id,
                        "upstream stream ended without [DONE] marker (policy=Required)"
                    );
                }
                StreamDoneOutcome::UnexpectedDoneMarker => {
                    tracing::warn!(
                        bridge = bridge_name,
                        request_id = %request_id,
                        "upstream emitted [DONE] marker (policy=None)"
                    );
                }
            }
        }
    }
}

fn parse_stream_chunk(
    payload: &str,
    reasoning_path: Option<&str>,
) -> Result<OpenAiStreamChunk, BridgeError> {
    match reasoning_path {
        Some(path) => {
            let mut value: Value = serde_json::from_str(payload)
                .map_err(|e| BridgeError::UpstreamDecode(e.to_string()))?;
            extract_reasoning_field(&mut value, path);
            serde_json::from_value(value).map_err(|e| BridgeError::UpstreamDecode(e.to_string()))
        }
        None => {
            serde_json::from_str(payload).map_err(|e| BridgeError::UpstreamDecode(e.to_string()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_accepts_canonical_https_resource() {
        let r = AzureUpstreamRef::resolve("gpt4o-prod", Some("https://acme-west.openai.azure.com"))
            .unwrap();
        assert_eq!(r.resource, "acme-west");
        assert_eq!(r.deployment, "gpt4o-prod");
        assert_eq!(r.api_version, AzureUpstreamRef::DEFAULT_API_VERSION);
    }

    #[test]
    fn resolve_accepts_bare_resource_name() {
        let r = AzureUpstreamRef::resolve("dep", Some("acme-east")).unwrap();
        assert_eq!(r.resource, "acme-east");
    }

    #[test]
    fn resolve_rejects_empty_deployment() {
        let err = AzureUpstreamRef::resolve("", Some("https://acme.openai.azure.com")).unwrap_err();
        match err {
            BridgeError::Config(msg) => {
                assert!(
                    msg.contains("deployment name is empty"),
                    "must call out empty deployment; got {msg}"
                );
            }
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    #[test]
    fn resolve_rejects_missing_api_base() {
        let err = AzureUpstreamRef::resolve("dep", None).unwrap_err();
        match err {
            BridgeError::Config(msg) => {
                assert!(
                    msg.contains("no api_base"),
                    "must call out missing api_base; got {msg}"
                );
            }
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    #[test]
    fn resolve_rejects_empty_api_base() {
        let err = AzureUpstreamRef::resolve("dep", Some("   ")).unwrap_err();
        match err {
            BridgeError::Config(msg) => {
                assert!(msg.contains("no api_base"));
            }
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    #[test]
    fn chat_completions_url_matches_azure_api_path() {
        let r = AzureUpstreamRef {
            resource: "acme-west".into(),
            deployment: "gpt4o-prod".into(),
            api_version: "2024-10-21".into(),
        };
        assert_eq!(
            r.chat_completions_url(),
            "https://acme-west.openai.azure.com/openai/deployments/gpt4o-prod/chat/completions?api-version=2024-10-21",
        );
    }

    #[test]
    fn resolve_rejects_deployment_with_query_injection() {
        let err = AzureUpstreamRef::resolve("foo?api-version=evil", Some("acme-east")).unwrap_err();
        match err {
            BridgeError::Config(msg) => {
                assert!(msg.contains("URL-control characters"), "got {msg}");
            }
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    #[test]
    fn resolve_rejects_deployment_with_slash_injection() {
        let err = AzureUpstreamRef::resolve("foo/bar/chat", Some("acme")).unwrap_err();
        assert!(matches!(err, BridgeError::Config(_)));
    }

    #[test]
    fn resolve_rejects_deployment_with_hash_fragment() {
        let err = AzureUpstreamRef::resolve("foo#bar", Some("acme")).unwrap_err();
        assert!(matches!(err, BridgeError::Config(_)));
    }

    #[test]
    fn resolve_rejects_resource_with_query_injection() {
        let err = AzureUpstreamRef::resolve("dep", Some("acme?evil=1")).unwrap_err();
        assert!(matches!(err, BridgeError::Config(_)));
    }

    #[test]
    fn resolve_rejects_canonical_https_with_wrong_suffix() {
        let err = AzureUpstreamRef::resolve("dep", Some("https://acme.evil.com")).unwrap_err();
        match err {
            BridgeError::Config(msg) => {
                assert!(
                    msg.contains("openai.azure.com"),
                    "must call out the required host suffix; got {msg}"
                );
            }
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    #[test]
    fn resolve_accepts_canonical_https_with_trailing_slash() {
        let r =
            AzureUpstreamRef::resolve("gpt4o-prod", Some("https://acme-west.openai.azure.com/"))
                .unwrap();
        assert_eq!(r.resource, "acme-west");
    }

    #[test]
    fn resolve_accepts_canonical_https_with_pasted_endpoint_path() {
        let r = AzureUpstreamRef::resolve(
            "gpt4o-prod",
            Some("https://acme-west.openai.azure.com/openai/deployments/x/chat/completions"),
        )
        .unwrap();
        assert_eq!(r.resource, "acme-west");
    }

    #[test]
    fn default_api_version_is_ga_shape() {
        let v = AzureUpstreamRef::DEFAULT_API_VERSION;
        assert!(
            !v.contains("preview"),
            "default API version must be GA, not preview; got {v:?}"
        );
        assert_eq!(v.len(), 10, "must match YYYY-MM-DD; got {v:?}");
        assert_eq!(v.chars().nth(4), Some('-'), "{v:?}");
        assert_eq!(v.chars().nth(7), Some('-'), "{v:?}");
    }

    #[test]
    fn bridge_name_is_stable() {
        assert_eq!(AzureOpenAiBridge::new().name(), "azure-openai");
    }

    // ─── Dispatch tests (wiremock) ────────────────────────────────────

    use aisix_core::{Model, ProviderKey};
    use aisix_gateway::ChatMessage;
    use std::sync::Arc;
    use wiremock::matchers::{header, method, path, query_param};
    use wiremock::{Mock, MockServer, Request as MockRequest, Respond, ResponseTemplate};

    /// Build a `BridgeContext` that points at a wiremock server. The
    /// server pretends to be `acme-west.openai.azure.com` — to make the
    /// reqwest client send there instead of the real Azure host we
    /// patch `chat_completions_url()` by routing through the mock host
    /// directly (see [`bridge_pointed_at_mock`]).
    fn sample_model() -> Arc<Model> {
        Arc::new(
            serde_json::from_str(
                r#"{
                    "display_name": "my-azure-gpt4",
                    "provider": "openai",
                    "model_name": "gpt4o-prod",
                    "provider_key_id": "11111111-1111-1111-1111-111111111111"
                }"#,
            )
            .unwrap(),
        )
    }

    fn sample_pk(api_base: Option<&str>) -> Arc<ProviderKey> {
        let api_base_json = match api_base {
            Some(b) => format!(r#", "api_base": "{b}""#),
            None => String::new(),
        };
        Arc::new(
            serde_json::from_str(&format!(
                r#"{{"display_name": "azure-prod", "secret": "az-key"{api_base_json}}}"#
            ))
            .unwrap(),
        )
    }

    /// Build a `ProviderKey` whose `secret` is the JSON-encoded AAD
    /// credentials shape. Used by the D6.6 (Entra ID) tests; sidesteps
    /// the string-escaping awkwardness of embedding JSON inside JSON.
    fn sample_pk_with_aad_secret(api_base: &str) -> Arc<ProviderKey> {
        let aad_json = r#"{"tenant_id":"tenant-uuid-aaa","client_id":"client-uuid-bbb","client_secret":"aad-secret-rotation-managed"}"#;
        let pk = serde_json::from_value::<ProviderKey>(serde_json::json!({
            "display_name": "azure-aad-prod",
            "secret": aad_json,
            "api_base": api_base,
        }))
        .unwrap();
        Arc::new(pk)
    }

    fn sample_pk_with_overrides(api_base: &str, overrides_json: &str) -> Arc<ProviderKey> {
        Arc::new(
            serde_json::from_str(&format!(
                r#"{{"display_name": "azure-prod", "secret": "az-key", "api_base": "{api_base}", {overrides_json}}}"#
            ))
            .unwrap(),
        )
    }

    /// Build a `BridgeContext` configured for dispatch tests:
    /// `Model.model_name = "gpt4o-prod"` and `ProviderKey.api_base`
    /// pinned to the canonical `https://acme-west.openai.azure.com`
    /// (so `AzureUpstreamRef::resolve` succeeds against the strict
    /// host-suffix check). The actual POST URL is rewritten by
    /// [`AzureOpenAiBridge::with_url_override`] to point at the
    /// wiremock server, so the test exercises the full `chat()` /
    /// `chat_stream()` entry point.
    fn canonical_test_ctx() -> BridgeContext {
        BridgeContext::new(
            "req-azure-1",
            sample_model(),
            sample_pk(Some("https://acme-west.openai.azure.com")),
        )
    }

    /// Compute the URL the wiremock server should receive — mirrors
    /// what `AzureUpstreamRef::chat_completions_url()` would produce
    /// but rooted at the mock's URI. Pass to
    /// [`AzureOpenAiBridge::with_url_override`].
    /// Test helper: build an [`AzureAuth`] for the api-key branch.
    /// Mirrors the inbound shape the production parser produces for
    /// a verbatim-string secret.
    fn api_key_auth(key: &str) -> AzureAuth {
        AzureAuth {
            api_key: Some(key.to_string()),
            bearer_token: None,
        }
    }

    fn mock_chat_url(mock_uri: &str, deployment: &str) -> String {
        format!(
            "{}/openai/deployments/{}/chat/completions?api-version=2024-10-21",
            mock_uri, deployment,
        )
    }

    #[test]
    fn build_request_headers_uses_api_key_not_bearer() {
        // Critical Azure-vs-OpenAI distinction: the auth header is
        // literally `api-key`, NOT `Authorization: Bearer`.
        let headers =
            build_request_headers(&api_key_auth("az-secret-key"), "req-1", false, None).unwrap();
        assert_eq!(headers.get("api-key").unwrap(), "az-secret-key");
        assert!(
            !headers.contains_key("authorization"),
            "must NOT set Authorization header for Azure api-key scheme"
        );
        assert_eq!(headers.get("content-type").unwrap(), "application/json");
        assert_eq!(headers.get("x-aisix-request-id").unwrap(), "req-1");
        assert!(
            !headers.contains_key("accept"),
            "Accept header must not be set for non-streaming requests"
        );
    }

    #[test]
    fn build_request_headers_sets_sse_accept_when_streaming() {
        let headers = build_request_headers(&api_key_auth("az-key"), "req-1", true, None).unwrap();
        assert_eq!(headers.get("accept").unwrap(), "text/event-stream");
    }

    #[test]
    fn build_request_headers_default_headers_cannot_override_api_key() {
        // Defense in depth: even if an operator's RequestOverrides
        // includes `default_headers.api-key`, the apply pipeline's
        // reserved-headers list must block it. Otherwise an org admin
        // who set up a Provider Key could exfil API traffic through
        // any header rewrite.
        use std::collections::HashMap;
        let mut default_headers = HashMap::new();
        default_headers.insert("api-key".to_string(), "ATTACKER-KEY".to_string());
        default_headers.insert("authorization".to_string(), "Bearer ATTACKER".to_string());
        let request_overrides = RequestOverrides {
            param_renames: HashMap::new(),
            param_constraints: None,
            default_body_fields: Default::default(),
            default_headers,
        };
        let headers = build_request_headers(
            &api_key_auth("legit-key"),
            "req-1",
            false,
            Some(&request_overrides),
        )
        .unwrap();
        assert_eq!(
            headers.get("api-key").unwrap(),
            "legit-key",
            "reserved-headers list must prevent api-key override"
        );
        assert!(
            !headers.contains_key("authorization"),
            "Authorization must not be set at all for Azure"
        );
    }

    #[test]
    fn build_request_headers_default_headers_allow_custom_non_reserved() {
        use std::collections::HashMap;
        let mut default_headers = HashMap::new();
        default_headers.insert("x-custom-trace".to_string(), "trace-123".to_string());
        let request_overrides = RequestOverrides {
            param_renames: HashMap::new(),
            param_constraints: None,
            default_body_fields: Default::default(),
            default_headers,
        };
        let headers =
            build_request_headers(&api_key_auth("k"), "req-1", false, Some(&request_overrides))
                .unwrap();
        assert_eq!(headers.get("x-custom-trace").unwrap(), "trace-123");
    }

    #[test]
    fn build_request_headers_rejects_invalid_api_key_chars() {
        // A secret with a newline would let an operator inject extra
        // headers via the api-key value.
        let err = build_request_headers(&api_key_auth("legit\nx-evil: 1"), "req-1", false, None)
            .unwrap_err();
        assert!(matches!(err, BridgeError::Config(_)));
    }

    #[test]
    fn build_request_headers_rejects_invalid_request_id_chars() {
        let err =
            build_request_headers(&api_key_auth("legit"), "req\nbad", false, None).unwrap_err();
        assert!(matches!(err, BridgeError::Config(_)));
    }

    #[tokio::test]
    async fn chat_with_missing_api_base_errors_before_dispatch() {
        let bridge = AzureOpenAiBridge::new();
        let ctx = BridgeContext::new("req-1", sample_model(), sample_pk(None));
        let req = ChatFormat::new("customer-facing-name", vec![ChatMessage::user("hi")]);
        let err = bridge.chat(&req, &ctx).await.unwrap_err();
        match err {
            BridgeError::Config(msg) => {
                assert!(msg.contains("no api_base"), "got {msg}");
            }
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn chat_with_empty_secret_errors_before_dispatch() {
        let bridge = AzureOpenAiBridge::new();
        // Build a PK whose secret is empty.
        let pk: Arc<ProviderKey> = Arc::new(
            serde_json::from_str(
                r#"{"display_name": "azure-prod", "secret": "", "api_base": "https://acme-west.openai.azure.com"}"#,
            )
            .unwrap(),
        );
        let ctx = BridgeContext::new("req-1", sample_model(), pk);
        let req = ChatFormat::new("customer-facing-name", vec![ChatMessage::user("hi")]);
        let err = bridge.chat(&req, &ctx).await.unwrap_err();
        match err {
            BridgeError::Config(msg) => {
                assert!(msg.contains("secret is empty"), "got {msg}");
            }
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    /// Per-test responder that records both the inbound request body
    /// AND headers so tests can assert (a) the renamed key carries the
    /// original value (Audit M2), (b) `Authorization` is absent at the
    /// wire (Audit M3 — defense in depth atop the unit-tested
    /// `build_request_headers`), and (c) the full body shape (Audit M4).
    #[derive(Clone, Default)]
    struct CapturingResponder {
        captured_body: std::sync::Arc<std::sync::Mutex<Option<serde_json::Value>>>,
        captured_headers: std::sync::Arc<std::sync::Mutex<Option<http::HeaderMap>>>,
        response_template: std::sync::Arc<std::sync::Mutex<Option<ResponseTemplate>>>,
    }

    impl CapturingResponder {
        fn with_response(self, template: ResponseTemplate) -> Self {
            *self.response_template.lock().unwrap() = Some(template);
            self
        }
    }

    impl Respond for CapturingResponder {
        fn respond(&self, req: &MockRequest) -> ResponseTemplate {
            let body: serde_json::Value = serde_json::from_slice(&req.body).unwrap_or_default();
            *self.captured_body.lock().unwrap() = Some(body);
            *self.captured_headers.lock().unwrap() = Some(req.headers.clone());
            self.response_template
                .lock()
                .unwrap()
                .clone()
                .unwrap_or_else(|| {
                    ResponseTemplate::new(200).set_body_json(serde_json::json!({
                        "id": "x", "model": "gpt4o-prod", "choices": [{
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop"
                        }], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
                    }))
                })
        }
    }

    // ─── bridge.chat() end-to-end against wiremock via url_override ────
    //
    // These tests drive the **real** `AzureOpenAiBridge::chat()` /
    // `chat_stream()` entry point. The canonical api_base
    // `https://acme-west.openai.azure.com` lets `AzureUpstreamRef::resolve`
    // succeed (strict host-suffix check passes), and
    // [`AzureOpenAiBridge::with_url_override`] rewrites the POST URL
    // to the wiremock server. So everything the bridge does at runtime
    // (header building, body building, override apply, error mapping,
    // SSE decoding, deadline handling) is exercised; only the final
    // hostname is different from production.

    #[tokio::test]
    async fn chat_dispatch_sends_api_key_header_and_deployment_url() {
        let server = MockServer::start().await;
        // Mock asserts: POST + path with deployment + api-version
        // query + api-key header carrying the literal secret.
        let responder = CapturingResponder::default().with_response(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-azure-1",
                "model": "gpt4o-prod",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi from azure"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
            })),
        );
        Mock::given(method("POST"))
            .and(path("/openai/deployments/gpt4o-prod/chat/completions"))
            .and(query_param("api-version", "2024-10-21"))
            .and(header("api-key", "az-key"))
            .and(header("content-type", "application/json"))
            .respond_with(responder.clone())
            .expect(1)
            .mount(&server)
            .await;

        let bridge =
            AzureOpenAiBridge::new().with_url_override(mock_chat_url(&server.uri(), "gpt4o-prod"));
        let ctx = canonical_test_ctx();
        let req = ChatFormat::new("my-azure-gpt4", vec![ChatMessage::user("hi")]);
        let chat = bridge.chat(&req, &ctx).await.unwrap();
        assert_eq!(chat.message.content, "hi from azure");

        // Audit M3: assert `Authorization` is absent on the wire (not
        // just absent from the helper's output).
        let captured = responder.captured_headers.lock().unwrap().clone().unwrap();
        assert!(
            !captured.contains_key("authorization"),
            "Authorization must not be on the wire; headers={captured:?}"
        );
        assert_eq!(
            captured.get("api-key").and_then(|v| v.to_str().ok()),
            Some("az-key"),
            "api-key must reach the wire with the literal secret"
        );
    }

    #[tokio::test]
    async fn chat_body_full_shape_on_the_wire() {
        // Audit M4: assert the full body shape, not just the `model`
        // field — `messages` array present, `stream: false` for
        // non-streaming, content matches the request.
        let server = MockServer::start().await;
        let responder = CapturingResponder::default();
        Mock::given(method("POST"))
            .and(path("/openai/deployments/gpt4o-prod/chat/completions"))
            .respond_with(responder.clone())
            .expect(1)
            .mount(&server)
            .await;

        let bridge =
            AzureOpenAiBridge::new().with_url_override(mock_chat_url(&server.uri(), "gpt4o-prod"));
        let ctx = canonical_test_ctx();
        let req = ChatFormat::new("my-azure-gpt4", vec![ChatMessage::user("hi")]);
        bridge.chat(&req, &ctx).await.unwrap();

        let body = responder.captured_body.lock().unwrap().clone().unwrap();
        assert_eq!(
            body.get("model").and_then(|v| v.as_str()),
            Some("gpt4o-prod"),
            "body.model must = deployment name; got body={body}"
        );
        let messages = body.get("messages").and_then(|v| v.as_array()).unwrap();
        assert_eq!(messages.len(), 1, "exactly one message; got body={body}");
        assert_eq!(
            messages[0].get("role").and_then(|v| v.as_str()),
            Some("user")
        );
        assert_eq!(
            messages[0].get("content").and_then(|v| v.as_str()),
            Some("hi")
        );
        assert_eq!(
            body.get("stream").and_then(|v| v.as_bool()),
            Some(false),
            "stream: false for chat (non-streaming); got body={body}"
        );
    }

    #[tokio::test]
    async fn chat_tolerates_content_filter_results_in_response() {
        // Azure-specific: responses include `prompt_filter_results`
        // and `content_filter_results` blocks. The reused OpenAi
        // wire parsers must not blow up on these extension fields —
        // they don't set `deny_unknown_fields`, so serde discards.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/openai/deployments/gpt4o-prod/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-azure-cf",
                "model": "gpt4o-prod",
                "prompt_filter_results": [{
                    "prompt_index": 0,
                    "content_filter_results": {
                        "hate": {"filtered": false, "severity": "safe"},
                        "self_harm": {"filtered": false, "severity": "safe"},
                        "sexual": {"filtered": false, "severity": "safe"},
                        "violence": {"filtered": false, "severity": "safe"}
                    }
                }],
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "filtered ok"},
                    "finish_reason": "stop",
                    "content_filter_results": {
                        "hate": {"filtered": false, "severity": "safe"}
                    }
                }],
                "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7}
            })))
            .expect(1)
            .mount(&server)
            .await;

        let bridge =
            AzureOpenAiBridge::new().with_url_override(mock_chat_url(&server.uri(), "gpt4o-prod"));
        let ctx = canonical_test_ctx();
        let req = ChatFormat::new("my-azure-gpt4", vec![ChatMessage::user("hi")]);
        let chat = bridge.chat(&req, &ctx).await.unwrap();
        assert_eq!(chat.message.content, "filtered ok");
        assert_eq!(chat.usage.total_tokens, 7);
    }

    #[tokio::test]
    async fn chat_applies_param_renames_to_outbound_body() {
        // Audit M2: assert the renamed key carries the original VALUE,
        // not just that the key swap happened.
        let server = MockServer::start().await;
        let responder = CapturingResponder::default();
        Mock::given(method("POST"))
            .and(path("/openai/deployments/gpt4o-prod/chat/completions"))
            .respond_with(responder.clone())
            .expect(1)
            .mount(&server)
            .await;

        let overrides_json = r#""request": {"param_renames": {"max_tokens": "max_completion_tokens"}, "param_constraints": null, "default_body_fields": {}, "default_headers": {}}"#;
        let pk = sample_pk_with_overrides("https://acme-west.openai.azure.com", overrides_json);
        let ctx = BridgeContext::new("r", sample_model(), pk);
        let req: ChatFormat = serde_json::from_str(
            r#"{"model": "my-azure-gpt4", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 100}"#,
        )
        .unwrap();
        let bridge =
            AzureOpenAiBridge::new().with_url_override(mock_chat_url(&server.uri(), "gpt4o-prod"));
        bridge.chat(&req, &ctx).await.unwrap();

        let body = responder.captured_body.lock().unwrap().clone().unwrap();
        assert!(
            body.get("max_tokens").is_none(),
            "original max_tokens key must be gone; body={body}"
        );
        assert_eq!(
            body.get("max_completion_tokens").and_then(|v| v.as_u64()),
            Some(100),
            "renamed key must carry the original value of 100; body={body}"
        );
    }

    #[tokio::test]
    async fn chat_maps_upstream_400_to_canned_message_not_body_echo() {
        // Audit M1: the upstream error body may contain operator-
        // internal identifiers (deployment name, resource hostname).
        // The bridge must map to a canned status-keyed phrase and NOT
        // echo the upstream body verbatim.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/openai/deployments/gpt4o-prod/chat/completions"))
            .respond_with(ResponseTemplate::new(400).set_body_string(
                "The API deployment for 'gpt4o-prod' does not exist on resource 'acme-west'",
            ))
            .expect(1)
            .mount(&server)
            .await;
        let bridge =
            AzureOpenAiBridge::new().with_url_override(mock_chat_url(&server.uri(), "gpt4o-prod"));
        let ctx = canonical_test_ctx();
        let req = ChatFormat::new("my-azure-gpt4", vec![ChatMessage::user("hi")]);
        let err = bridge.chat(&req, &ctx).await.unwrap_err();
        match err {
            BridgeError::UpstreamStatus {
                status, message, ..
            } => {
                assert_eq!(status, 400);
                assert!(
                    !message.contains("gpt4o-prod") && !message.contains("acme-west"),
                    "upstream body must not echo into the customer-visible error; got message={message:?}"
                );
            }
            other => panic!("expected UpstreamStatus, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn chat_maps_404_to_deployment_not_found_canned_message() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/openai/deployments/gpt4o-prod/chat/completions"))
            .respond_with(ResponseTemplate::new(404).set_body_string("operator-internal: foo-bar"))
            .expect(1)
            .mount(&server)
            .await;
        let bridge =
            AzureOpenAiBridge::new().with_url_override(mock_chat_url(&server.uri(), "gpt4o-prod"));
        let ctx = canonical_test_ctx();
        let req = ChatFormat::new("my-azure-gpt4", vec![ChatMessage::user("hi")]);
        let err = bridge.chat(&req, &ctx).await.unwrap_err();
        match err {
            BridgeError::UpstreamStatus {
                status, message, ..
            } => {
                assert_eq!(status, 404);
                assert!(
                    message.contains("deployment or model not found"),
                    "404 must surface as deployment-not-found canned message; got {message:?}"
                );
                assert!(
                    !message.contains("foo-bar"),
                    "upstream body must not leak; got {message:?}"
                );
            }
            other => panic!("expected UpstreamStatus, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn chat_maps_429_with_retry_after_and_canned_message() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/openai/deployments/gpt4o-prod/chat/completions"))
            .respond_with(
                ResponseTemplate::new(429)
                    .insert_header("retry-after", "30")
                    .set_body_string("rate limited — quota for deployment gpt4o-prod exceeded"),
            )
            .expect(1)
            .mount(&server)
            .await;
        let bridge =
            AzureOpenAiBridge::new().with_url_override(mock_chat_url(&server.uri(), "gpt4o-prod"));
        let ctx = canonical_test_ctx();
        let req = ChatFormat::new("my-azure-gpt4", vec![ChatMessage::user("hi")]);
        let err = bridge.chat(&req, &ctx).await.unwrap_err();
        match err {
            BridgeError::UpstreamStatus {
                status,
                retry_after,
                message,
                ..
            } => {
                assert_eq!(status, 429);
                assert_eq!(retry_after, Some(std::time::Duration::from_secs(30)));
                assert!(
                    !message.contains("gpt4o-prod"),
                    "upstream body must not leak; got {message:?}"
                );
            }
            other => panic!("expected UpstreamStatus with retry_after, got {other:?}"),
        }
    }

    /// Copilot review (PR #323): Azure's envelope only has `error.code`
    /// (no separate `type`). For OpenAI-compatible tokens that Azure
    /// inherits unchanged (e.g. `rate_limit_exceeded`), the bridge
    /// must populate BOTH `parsed.kind` AND `parsed.code` from the
    /// upstream — otherwise the downstream OpenAI client receives
    /// `error.type=rate_limit_exceeded` but `error.code=null`, which
    /// is exactly the SDK-retry-logic break that issue #322 calls out.
    #[tokio::test]
    async fn chat_429_preserves_openai_compatible_code_for_sdk_retry() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/openai/deployments/gpt4o-prod/chat/completions"))
            .respond_with(ResponseTemplate::new(429).set_body_raw(
                br#"{"error":{"code":"rate_limit_exceeded","message":"slow down"}}"#.as_slice(),
                "application/json",
            ))
            .mount(&server)
            .await;
        let bridge =
            AzureOpenAiBridge::new().with_url_override(mock_chat_url(&server.uri(), "gpt4o-prod"));
        let ctx = canonical_test_ctx();
        let req = ChatFormat::new("my-azure-gpt4", vec![ChatMessage::user("hi")]);
        let err = bridge.chat(&req, &ctx).await.unwrap_err();
        match err {
            BridgeError::UpstreamStatus { parsed, .. } => {
                let parsed = parsed.expect("envelope parsed");
                assert_eq!(parsed.kind.as_deref(), Some("rate_limit_exceeded"));
                assert_eq!(
                    parsed.code.as_deref(),
                    Some("rate_limit_exceeded"),
                    "OpenAI-compat code must flow through view.code for SDK retry"
                );
            }
            other => panic!("expected UpstreamStatus, got {other:?}"),
        }
    }

    /// Copilot review (PR #323): when upstream returns a non-JSON
    /// body (HTML error page from Azure Front Door, etc.), the parser
    /// must skip the serde attempt rather than try to deserialize
    /// `<html>...</html>` as JSON. Same content-type guard as
    /// `capture_upstream_error_http`.
    #[tokio::test]
    async fn chat_400_non_json_body_skips_envelope_parse() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/openai/deployments/gpt4o-prod/chat/completions"))
            .respond_with(ResponseTemplate::new(400).set_body_raw(
                b"<html><body>403 Forbidden by Front Door</body></html>".as_slice(),
                "text/html",
            ))
            .mount(&server)
            .await;
        let bridge =
            AzureOpenAiBridge::new().with_url_override(mock_chat_url(&server.uri(), "gpt4o-prod"));
        let ctx = canonical_test_ctx();
        let req = ChatFormat::new("my-azure-gpt4", vec![ChatMessage::user("hi")]);
        let err = bridge.chat(&req, &ctx).await.unwrap_err();
        match err {
            BridgeError::UpstreamStatus { parsed, .. } => {
                assert!(
                    parsed.is_none(),
                    "non-JSON body must not produce a parsed view; got {parsed:?}"
                );
            }
            other => panic!("expected UpstreamStatus, got {other:?}"),
        }
    }

    /// Audit fix (PR #323 follow-up): the `inner_error` casing
    /// variant is what most Azure docs show, but Azure ALSO emits
    /// `innererror` (smushed) on some endpoints. Both must be
    /// recognised by the parser.
    #[tokio::test]
    async fn chat_400_with_innererror_smushed_casing_also_lifts_kind() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/openai/deployments/gpt4o-prod/chat/completions"))
            .respond_with(ResponseTemplate::new(400).set_body_raw(
                br#"{"error":{"code":"invalid_request_error","message":"blocked","innererror":{"code":"ResponsibleAIPolicyViolation"}}}"#.as_slice(),
                "application/json",
            ))
            .mount(&server)
            .await;
        let bridge =
            AzureOpenAiBridge::new().with_url_override(mock_chat_url(&server.uri(), "gpt4o-prod"));
        let ctx = canonical_test_ctx();
        let req = ChatFormat::new("my-azure-gpt4", vec![ChatMessage::user("hi")]);
        let err = bridge.chat(&req, &ctx).await.unwrap_err();
        match err {
            BridgeError::UpstreamStatus { parsed, .. } => {
                let parsed = parsed.expect("innererror parsed");
                assert_eq!(parsed.kind.as_deref(), Some("ResponsibleAIPolicyViolation"));
            }
            other => panic!("expected UpstreamStatus, got {other:?}"),
        }
    }

    /// Audit fix (PR #323 MEDIUM-2): structured-parse path —
    /// Azure-specific `inner_error.code = ResponsibleAIPolicyViolation`
    /// must surface as `parsed.kind`, not be flattened under the outer
    /// `error.code`. `wire: AzureOpenAI` must be set so the
    /// translation layer picks the Azure-aware code map.
    /// `parsed.message` stays `None` (deployment id leak).
    #[tokio::test]
    async fn chat_400_with_inner_error_responsible_ai_lifts_kind() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/openai/deployments/gpt4o-prod/chat/completions"))
            .respond_with(ResponseTemplate::new(400).set_body_raw(
                br#"{"error":{"code":"invalid_request_error","message":"blocked","inner_error":{"code":"ResponsibleAIPolicyViolation"}}}"#.as_slice(),
                "application/json",
            ))
            .mount(&server)
            .await;
        let bridge =
            AzureOpenAiBridge::new().with_url_override(mock_chat_url(&server.uri(), "gpt4o-prod"));
        let ctx = canonical_test_ctx();
        let req = ChatFormat::new("my-azure-gpt4", vec![ChatMessage::user("hi")]);
        let err = bridge.chat(&req, &ctx).await.unwrap_err();
        match err {
            BridgeError::UpstreamStatus {
                status,
                wire,
                parsed,
                ..
            } => {
                assert_eq!(status, 400);
                assert_eq!(wire, aisix_gateway::UpstreamWire::AzureOpenAI);
                let parsed = parsed.expect("inner_error parsed");
                assert_eq!(parsed.kind.as_deref(), Some("ResponsibleAIPolicyViolation"));
                assert!(parsed.message.is_none());
            }
            other => panic!("expected UpstreamStatus, got {other:?}"),
        }
    }

    /// Audit L3: `chat_against_full_bridge_dispatch` calls the real
    /// Azure DNS (`acme-west.openai.azure.com`). Marked `#[ignore]`
    /// because (a) CI runners with corporate proxies may resolve it
    /// to unexpected hosts, (b) it takes wall-clock time to fail. Run
    /// manually with `cargo test -- --ignored` to sanity-check that
    /// the bridge reaches the network layer end-to-end.
    #[tokio::test]
    #[ignore = "calls real Azure DNS; run with `cargo test -- --ignored`"]
    async fn chat_against_real_azure_reaches_network() {
        let bridge = AzureOpenAiBridge::new();
        let ctx = BridgeContext::new(
            "req-1",
            sample_model(),
            sample_pk(Some("https://acme-west.openai.azure.com")),
        );
        let req = ChatFormat::new("my-azure-gpt4", vec![ChatMessage::user("hi")]);
        let err = bridge.chat(&req, &ctx).await.unwrap_err();
        match err {
            BridgeError::Transport(_) | BridgeError::UpstreamStatus { .. } => {
                // expected — we reached the network
            }
            other => panic!(
                "expected Transport or UpstreamStatus (proving we reached network); got {other:?}"
            ),
        }
    }

    /// D6 audit HIGH-1 regression (from #313 skeleton): dispatch must
    /// read the upstream deployment from `ctx.model.model_name`, NOT
    /// from `req.model`. `req.model` is the customer-typed display
    /// name; resolving off it would produce
    /// `/openai/deployments/customer-facing-name/...` — 404 from
    /// Azure every time.
    #[tokio::test]
    async fn chat_ignores_req_model_and_uses_ctx_model_name() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/openai/deployments/gpt4o-prod/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "x", "model": "gpt4o-prod", "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop"
                }], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            })))
            .expect(1)
            .mount(&server)
            .await;

        let bridge =
            AzureOpenAiBridge::new().with_url_override(mock_chat_url(&server.uri(), "gpt4o-prod"));
        let ctx = canonical_test_ctx();
        // req.model is the customer-facing display name. The URL the
        // bridge dispatches to must use ctx.model.model_name
        // ("gpt4o-prod") not req.model ("customer-facing-name").
        let req = ChatFormat::new("customer-facing-name", vec![ChatMessage::user("hi")]);
        bridge.chat(&req, &ctx).await.unwrap();
        // Path matcher on `gpt4o-prod` proves dispatch used model_name.
    }

    #[tokio::test]
    async fn chat_stream_yields_chunks_until_done_marker_with_inline_content_filters() {
        // Audit H3: SSE stream chunks can carry `prompt_filter_results`
        // (top-level) and `content_filter_results` (per-choice) — the
        // reused OpenAiStreamChunk parsers must tolerate both without
        // breaking deserialization.
        let server = MockServer::start().await;
        let sse_body = concat!(
            // chunk 1 — top-level prompt_filter_results + empty choices (Azure prelude)
            "data: {\"id\":\"x\",\"model\":\"gpt4o-prod\",\"prompt_filter_results\":[{\"prompt_index\":0,\"content_filter_results\":{\"hate\":{\"filtered\":false,\"severity\":\"safe\"}}}],\"choices\":[]}\n\n",
            // chunk 2 — content delta with per-choice content_filter_results
            "data: {\"id\":\"x\",\"model\":\"gpt4o-prod\",\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"hello\"},\"content_filter_results\":{\"hate\":{\"filtered\":false,\"severity\":\"safe\"}},\"finish_reason\":null}]}\n\n",
            // chunk 3 — content delta with finish_reason
            "data: {\"id\":\"x\",\"model\":\"gpt4o-prod\",\"choices\":[{\"delta\":{\"content\":\" world\"},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n",
        );
        Mock::given(method("POST"))
            .and(path("/openai/deployments/gpt4o-prod/chat/completions"))
            .and(header("accept", "text/event-stream"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse_body),
            )
            .expect(1)
            .mount(&server)
            .await;

        let bridge =
            AzureOpenAiBridge::new().with_url_override(mock_chat_url(&server.uri(), "gpt4o-prod"));
        let ctx = canonical_test_ctx();
        let req = ChatFormat::new("my-azure-gpt4", vec![ChatMessage::user("hi")]);
        let mut stream = bridge.chat_stream(&req, &ctx).await.unwrap();
        let mut chunks = Vec::new();
        while let Some(item) = stream.next().await {
            chunks.push(item.unwrap());
        }
        // Audit H3 specifically: chunk 1 has empty choices — yielded
        // chunk's delta is the default. Chunk 2 carries "hello".
        // Chunk 3 carries " world" + finish_reason=stop.
        assert!(
            chunks.len() >= 3,
            "expected at least 3 chunks (incl. Azure prelude); got {chunks:?}"
        );
        let content_chunk = chunks
            .iter()
            .find(|c| c.delta.content.as_deref() == Some("hello"))
            .expect("must find a chunk with content=hello");
        assert_eq!(
            content_chunk.delta.role,
            Some(aisix_gateway::Role::Assistant)
        );
        let last = chunks.last().unwrap();
        assert!(
            last.finish_reason.is_some(),
            "last chunk must carry finish_reason"
        );
    }

    #[tokio::test]
    async fn chat_stream_enforces_per_chunk_deadline() {
        // Audit H2: a slow / hanging stream body must not wedge after
        // headers arrive. The bridge enforces the deadline on each
        // per-chunk wait, not just on the initial POST.
        let server = MockServer::start().await;
        // 1s delay before the SSE body is emitted. With a 200ms
        // deadline, the bridge should surface a Timeout from the
        // per-chunk wait — the headers arrive instantly (mock
        // responds with 200), but the body delivery sits idle.
        Mock::given(method("POST"))
            .and(path("/openai/deployments/gpt4o-prod/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_delay(std::time::Duration::from_secs(2))
                    .set_body_string("data: [DONE]\n\n"),
            )
            .mount(&server)
            .await;

        let bridge =
            AzureOpenAiBridge::new().with_url_override(mock_chat_url(&server.uri(), "gpt4o-prod"));
        let mut ctx = canonical_test_ctx();
        ctx.deadline = Some(std::time::Duration::from_millis(200));
        let req = ChatFormat::new("my-azure-gpt4", vec![ChatMessage::user("hi")]);
        // The initial POST happens within deadline (mock 200 OK), but
        // wiremock's set_delay applies to the response *body*, so
        // either the initial with_deadline or the per-chunk timeout
        // fires. Either way the bridge must emit a Timeout.
        let result = bridge.chat_stream(&req, &ctx).await;
        match result {
            Ok(mut stream) => {
                let next = stream.next().await;
                match next {
                    Some(Err(BridgeError::Timeout { .. })) => {}
                    other => panic!("expected per-chunk Timeout, got {other:?}"),
                }
            }
            Err(BridgeError::Timeout { .. }) => {
                // Acceptable: initial POST already timed out.
            }
            Err(other) => panic!("expected Timeout, got {other:?}"),
        }
    }

    // ─── AAD (Entra ID) auth scheme (D6.6) ─────────────────────────

    /// JSON-shaped Azure secret triggering the AAD branch.
    fn aad_secret_json() -> String {
        r#"{
            "tenant_id": "tenant-uuid-aaa",
            "client_id": "client-uuid-bbb",
            "client_secret": "aad-secret-rotation-managed"
        }"#
        .to_string()
    }

    #[test]
    fn azure_secret_parses_verbatim_string_as_api_key() {
        // Backward compat: existing pre-D6.6 deployments encode the
        // resource api-key as a bare string.
        let parsed = AzureSecret::parse("legacy-api-key-string").unwrap();
        match parsed {
            AzureSecret::ApiKey(k) => assert_eq!(k, "legacy-api-key-string"),
            AzureSecret::Aad(_) => panic!("verbatim string must parse as api-key, not AAD"),
        }
    }

    #[test]
    fn azure_secret_parses_json_object_as_aad_credentials() {
        let parsed = AzureSecret::parse(&aad_secret_json()).unwrap();
        match parsed {
            AzureSecret::Aad(creds) => {
                assert_eq!(creds.tenant_id, "tenant-uuid-aaa");
                assert_eq!(creds.client_id, "client-uuid-bbb");
                assert_eq!(creds.client_secret, "aad-secret-rotation-managed");
            }
            AzureSecret::ApiKey(_) => panic!("JSON-shaped secret must parse as AAD, not api-key"),
        }
    }

    #[test]
    fn azure_secret_rejects_empty_secret() {
        let err = AzureSecret::parse("   ").unwrap_err();
        assert!(matches!(err, BridgeError::Config(_)));
    }

    #[test]
    fn azure_secret_rejects_json_missing_required_aad_fields() {
        let err = AzureSecret::parse(r#"{"tenant_id":"t"}"#).unwrap_err();
        match err {
            BridgeError::Config(msg) => {
                // Message must NOT echo raw secret bytes (audit-aware).
                assert!(msg.contains("looks JSON-shaped"));
                assert!(!msg.contains("tenant-uuid-aaa"));
            }
            other => panic!("expected Config, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn chat_with_aad_secret_mints_token_and_sets_authorization_bearer_header() {
        // End-to-end pin: AAD JSON secret → token mint via wiremock
        // AAD endpoint → chat POST to Azure OpenAI endpoint carrying
        // Authorization: Bearer <minted-token> (NOT api-key:).
        let aad_server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/oauth2/v2.0/token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "access_token": "aad.bearer.test-token",
                "expires_in": 3600,
                "token_type": "Bearer"
            })))
            .expect(1)
            .mount(&aad_server)
            .await;

        let azure_server = MockServer::start().await;
        let captured_headers: std::sync::Arc<std::sync::Mutex<Option<http::HeaderMap>>> =
            std::sync::Arc::new(std::sync::Mutex::new(None));
        let captured_for_responder = captured_headers.clone();
        Mock::given(method("POST"))
            .respond_with(move |req: &wiremock::Request| {
                *captured_for_responder.lock().unwrap() = Some(req.headers.clone());
                ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "id": "chatcmpl-aad-test",
                    "object": "chat.completion",
                    "created": 1700000000_i64,
                    "model": "gpt-4o",
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": "hello via aad"},
                        "finish_reason": "stop"
                    }],
                    "usage": {"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10}
                }))
            })
            .mount(&azure_server)
            .await;

        let bridge = AzureOpenAiBridge::new()
            .with_url_override(mock_chat_url(&azure_server.uri(), "gpt4o-prod"))
            .with_aad_token_endpoint_override(format!("{}/oauth2/v2.0/token", aad_server.uri()));
        let ctx = BridgeContext::new(
            "req-azure-1",
            sample_model(),
            sample_pk_with_aad_secret("https://acme-west.openai.azure.com"),
        );
        let req = ChatFormat::new("my-azure-gpt4", vec![ChatMessage::user("hi")]);
        let resp = bridge.chat(&req, &ctx).await.unwrap();
        assert_eq!(resp.message.content, "hello via aad");

        let headers = captured_headers
            .lock()
            .unwrap()
            .clone()
            .expect("headers captured");
        assert_eq!(
            headers.get("authorization").unwrap(),
            "Bearer aad.bearer.test-token",
            "AAD path must send Authorization: Bearer <minted-token>"
        );
        assert!(
            !headers.contains_key("api-key"),
            "AAD path must NOT send api-key: header (mutually exclusive auth schemes)"
        );
    }

    #[tokio::test]
    async fn chat_with_aad_secret_caches_minted_token_across_calls() {
        // Three back-to-back chats with the same AAD secret must
        // share the cache slot — only one token-mint trip.
        let aad_server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/oauth2/v2.0/token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "access_token": "aad.cached-token",
                "expires_in": 3600,
                "token_type": "Bearer"
            })))
            .expect(1) // critical: must be called EXACTLY ONCE across 3 chats
            .mount(&aad_server)
            .await;

        let azure_server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "x", "object": "chat.completion", "created": 1700000000_i64,
                "model": "gpt-4o",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            })))
            .mount(&azure_server)
            .await;

        let bridge = AzureOpenAiBridge::new()
            .with_url_override(mock_chat_url(&azure_server.uri(), "gpt4o-prod"))
            .with_aad_token_endpoint_override(format!("{}/oauth2/v2.0/token", aad_server.uri()));
        let ctx = BridgeContext::new(
            "req-azure-1",
            sample_model(),
            sample_pk_with_aad_secret("https://acme-west.openai.azure.com"),
        );
        let req = ChatFormat::new("my-azure-gpt4", vec![ChatMessage::user("hi")]);
        for _ in 0..3 {
            bridge.chat(&req, &ctx).await.unwrap();
        }
        // wiremock's `expect(1)` on the AAD mock asserts on server
        // drop that only one mint trip happened across three chats.
    }

    #[tokio::test]
    async fn chat_with_aad_secret_aad_4xx_surfaces_before_azure_call() {
        // AAD 401 bubbles out as Config (operator-actionable) without
        // ever calling the Azure OpenAI endpoint.
        let aad_server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(401).set_body_string(
                r#"{"error":"invalid_client","error_description":"AADSTS7000215"}"#,
            ))
            .mount(&aad_server)
            .await;

        let azure_server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200))
            .expect(0) // must NOT be called
            .mount(&azure_server)
            .await;

        let bridge = AzureOpenAiBridge::new()
            .with_url_override(mock_chat_url(&azure_server.uri(), "gpt4o-prod"))
            .with_aad_token_endpoint_override(format!("{}/oauth2/v2.0/token", aad_server.uri()));
        let ctx = BridgeContext::new(
            "req-azure-1",
            sample_model(),
            sample_pk_with_aad_secret("https://acme-west.openai.azure.com"),
        );
        let req = ChatFormat::new("my-azure-gpt4", vec![ChatMessage::user("hi")]);
        let err = bridge.chat(&req, &ctx).await.unwrap_err();
        match err {
            BridgeError::Config(msg) => {
                assert!(msg.contains("invalid_client"));
            }
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    /// Audit LOW (audit-aigw-388-azure-aad): chat_stream must also
    /// resolve auth via the AAD path. The streaming code calls the
    /// same `resolve_auth` helper as chat(), so this regression-
    /// guards a future refactor that accidentally skipped it.
    /// Mirrors the same gap noted (and addressed in follow-up) by
    /// audit-aigw-387 on the Vertex SA OAuth side.
    #[tokio::test]
    async fn chat_stream_with_aad_secret_sets_authorization_bearer_header() {
        let aad_server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/oauth2/v2.0/token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "access_token": "aad.stream-bearer",
                "expires_in": 3600,
                "token_type": "Bearer"
            })))
            .expect(1)
            .mount(&aad_server)
            .await;

        let azure_server = MockServer::start().await;
        let captured_headers: std::sync::Arc<std::sync::Mutex<Option<http::HeaderMap>>> =
            std::sync::Arc::new(std::sync::Mutex::new(None));
        let captured_for_responder = captured_headers.clone();
        Mock::given(method("POST"))
            .respond_with(move |req: &wiremock::Request| {
                *captured_for_responder.lock().unwrap() = Some(req.headers.clone());
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string("data: [DONE]\n\n")
            })
            .mount(&azure_server)
            .await;

        let bridge = AzureOpenAiBridge::new()
            .with_url_override(mock_chat_url(&azure_server.uri(), "gpt4o-prod"))
            .with_aad_token_endpoint_override(format!("{}/oauth2/v2.0/token", aad_server.uri()));
        let ctx = BridgeContext::new(
            "req-azure-1",
            sample_model(),
            sample_pk_with_aad_secret("https://acme-west.openai.azure.com"),
        );
        let req = ChatFormat::new("my-azure-gpt4", vec![ChatMessage::user("hi")]);
        let mut stream = bridge.chat_stream(&req, &ctx).await.unwrap();
        while stream.next().await.is_some() {}

        let headers = captured_headers
            .lock()
            .unwrap()
            .clone()
            .expect("headers captured");
        assert_eq!(
            headers.get("authorization").unwrap(),
            "Bearer aad.stream-bearer",
            "chat_stream AAD path must send Authorization: Bearer <minted-token>"
        );
        assert!(
            !headers.contains_key("api-key"),
            "chat_stream AAD path must NOT send api-key: header"
        );
        assert_eq!(
            headers.get("accept").unwrap(),
            "text/event-stream",
            "chat_stream must request SSE via Accept header"
        );
    }
}
