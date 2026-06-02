//! `AnthropicBridge` — concrete [`Bridge`] for the Claude Messages API.
//!
//! Mirrors `OpenAiBridge`'s transport shape but differs in three important
//! places:
//!
//! - **Auth header**: `x-api-key: <key>` + `anthropic-version` (Bearer not
//!   accepted).
//! - **Endpoint**: `POST {base}/v1/messages`. We append `/v1/messages`
//!   ourselves because the Model's `api_base` is the host, not the
//!   messages endpoint.
//! - **Stream model**: event-typed SSE where only a couple of variants
//!   yield user-visible chunks. We drive that via `StreamState`.
//!
//! Error mapping is identical to OpenAi — the `BridgeError` contract from
//! PR #6 applies verbatim.

use aisix_gateway::{
    Bridge, BridgeContext, BridgeError, ChatChunk, ChatChunkStream, ChatFormat, ChatResponse,
    SseDecoder, SseEvent,
};
use async_trait::async_trait;
use futures::StreamExt;
use reqwest::{header, Client, StatusCode};
use std::time::{Duration, Instant};

use crate::wire::{
    build_request, response_into_chat_response, split_system, AnthropicResponse,
    AnthropicStreamEvent, StreamState,
};

/// Matches the API header that Anthropic bakes backwards-compat into.
/// Pinned here rather than config-driven so each bridge version ships
/// a known compatible version string; bumping it is a code change.
pub const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Fallback host used when the Model doesn't set `api_base` and the
/// Provider enum's default is missing. Real operators set `api_base`
/// on the Model to point at the Anthropic-owned endpoint they use.
pub const ANTHROPIC_DEFAULT_BASE: &str = "https://api.anthropic.com";

pub struct AnthropicBridge {
    client: Client,
    api_version: &'static str,
}

impl AnthropicBridge {
    pub fn new() -> Self {
        Self::with_client(default_client())
    }

    pub fn with_client(client: Client) -> Self {
        Self {
            client,
            api_version: ANTHROPIC_VERSION,
        }
    }

    pub fn with_api_version(mut self, v: &'static str) -> Self {
        self.api_version = v;
        self
    }
}

impl Default for AnthropicBridge {
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

/// Path suffixes the Anthropic bridge appends. If an operator
/// accidentally pastes a fuller form into `api_base`, strip the suffix
/// so request building still produces the right URL.
const ANTHROPIC_ENDPOINT_SUFFIXES: &[&str] = &["/v1/messages", "/v1"];

/// Tolerate the common variations of `api_base` an operator might
/// paste for the Anthropic upstream. Accepted forms:
///
/// - `https://api.anthropic.com` (canonical)
/// - `https://api.anthropic.com/` (trailing slash)
/// - `https://api.anthropic.com/v1` (extra `/v1` segment — common copy-paste
///   habit from OpenAI conventions)
/// - `https://api.anthropic.com/v1/messages` (full upstream URL pasted)
///
/// All collapse to the canonical bare host. The bridge then appends
/// `/v1/messages` at request time.
fn normalize_api_base(base: &str) -> String {
    let trimmed = base.trim_end_matches('/');
    for suffix in ANTHROPIC_ENDPOINT_SUFFIXES {
        if let Some(rest) = trimmed.strip_suffix(suffix) {
            return rest.trim_end_matches('/').to_string();
        }
    }
    trimmed.to_string()
}

fn resolve_base(ctx: &BridgeContext) -> Result<String, BridgeError> {
    match ctx.provider_key.api_base.as_deref() {
        Some(b) if !b.trim().is_empty() => Ok(normalize_api_base(b.trim())),
        _ => {
            // Family-bridge safety: when `AnthropicBridge` is the
            // family registration (registered via
            // `register_family(Adapter::Anthropic, ...)`) AND the
            // dispatching `ProviderKey.provider` identifies a vendor
            // that ISN'T anthropic, refuse to fall back to
            // `ANTHROPIC_DEFAULT_BASE` — that would silently route the
            // vendor's API key to `api.anthropic.com`. Mirrors the
            // OpenAI-family guard in
            // `crates/aisix-provider-openai/src/bridge.rs::resolve_base`.
            //
            // Pre-Phase-A rows (empty `ProviderKey.provider`) fall
            // through to the historical default-base path unchanged.
            let pk_vendor_raw = ctx.provider_key.provider.as_str();
            let pk_vendor_normalized = pk_vendor_raw.trim().to_ascii_lowercase();
            if !pk_vendor_normalized.is_empty() && pk_vendor_normalized != "anthropic" {
                // Operator-facing detail (route, provider topology,
                // remediation steps) goes to logs only — keep the
                // customer-visible 500 body short and free of
                // internal-product taxonomy (cp-api / adapter_map /
                // provider_metadata field names are not part of any
                // wire contract a customer should depend on).
                tracing::error!(
                    target: "aisix_provider_anthropic::bridge",
                    pk_display_name = %ctx.provider_key.display_name,
                    pk_vendor = %pk_vendor_raw,
                    "provider_key has no api_base; family bridge refusing fallback to \
                     api.anthropic.com. Operator action: populate `api_base` on the \
                     ProviderKey resource (managed deployments: via adapter_map / \
                     provider_metadata.api_base_url on the control plane; standalone: \
                     directly on the resource)."
                );
                return Err(BridgeError::InvalidUpstreamConfig(format!(
                    "provider_key for vendor {pk_vendor_raw:?} has no upstream base URL \
                     configured"
                )));
            }
            Ok(ANTHROPIC_DEFAULT_BASE.to_string())
        }
    }
}

fn api_key(ctx: &BridgeContext) -> Result<&str, BridgeError> {
    let k = &ctx.provider_key.secret;
    if k.is_empty() {
        return Err(BridgeError::InvalidUpstreamConfig(
            "provider_key.secret is empty".into(),
        ));
    }
    // Reject a secret that can't be a valid `x-api-key` header value
    // (control bytes etc.) up front as customer-fixable config, mirroring
    // the openai / azure bridges — otherwise reqwest's `.header()` fails
    // later with an opaque builder error (#367).
    if header::HeaderValue::from_str(k).is_err() {
        return Err(BridgeError::InvalidUpstreamConfig(
            "provider_key.secret contains invalid header characters".into(),
        ));
    }
    Ok(k.as_str())
}

fn upstream_model(ctx: &BridgeContext) -> Result<&str, BridgeError> {
    ctx.model
        .model_name
        .as_deref()
        .ok_or_else(|| BridgeError::InvalidUpstreamConfig("model.model_name missing".into()))
}

async fn map_http_error(status: StatusCode, resp: reqwest::Response) -> BridgeError {
    aisix_gateway::capture_upstream_error_http(
        status,
        resp,
        aisix_gateway::UpstreamWire::Anthropic,
        parse_anthropic_error_envelope,
    )
    .await
}

/// Parse the Anthropic error envelope:
///
/// ```json
/// {"type": "error", "error": {"type": "...", "message": "..."}}
/// ```
///
/// Anthropic does not carry `code` or `param` fields — those stay
/// `None`. The translation table at render time derives an OpenAI
/// `code` from `kind` when crossing wire formats.
///
/// Reference: <https://docs.anthropic.com/en/api/errors>
fn parse_anthropic_error_envelope(body: &[u8]) -> Option<aisix_gateway::UpstreamErrorView> {
    #[derive(serde::Deserialize)]
    struct Outer {
        error: Inner,
    }
    #[derive(serde::Deserialize)]
    struct Inner {
        #[serde(rename = "type")]
        kind: Option<String>,
        message: Option<String>,
    }
    let outer: Outer = serde_json::from_slice(body).ok()?;
    Some(aisix_gateway::UpstreamErrorView {
        kind: outer.error.kind,
        message: outer.error.message,
        code: None,
        param: None,
    })
}

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

#[async_trait]
impl Bridge for AnthropicBridge {
    fn name(&self) -> &'static str {
        "anthropic"
    }

    async fn chat(
        &self,
        req: &ChatFormat,
        ctx: &BridgeContext,
    ) -> Result<ChatResponse, BridgeError> {
        let base = resolve_base(ctx)?;
        let key = api_key(ctx)?;
        let upstream = upstream_model(ctx)?;

        let (system, messages) =
            split_system(req).map_err(|e| BridgeError::InvalidUpstreamConfig(e.to_string()))?;
        let body = build_request(req, upstream, system, messages, false);
        let url = format!("{base}/v1/messages");
        let client = self.client.clone();
        let api_version = self.api_version;
        let started = Instant::now();
        let request_id = ctx.request_id.clone();

        with_deadline(ctx.deadline, started, async move {
            let resp = client
                .post(&url)
                .header("x-api-key", key)
                .header("anthropic-version", api_version)
                .header(header::CONTENT_TYPE, "application/json")
                .header("x-aisix-request-id", &request_id)
                .json(&body)
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

    async fn chat_stream(
        &self,
        req: &ChatFormat,
        ctx: &BridgeContext,
    ) -> Result<ChatChunkStream, BridgeError> {
        let base = resolve_base(ctx)?;
        let key = api_key(ctx)?;
        let upstream = upstream_model(ctx)?;

        let (system, messages) =
            split_system(req).map_err(|e| BridgeError::InvalidUpstreamConfig(e.to_string()))?;
        let body = build_request(req, upstream, system, messages, true);
        let url = format!("{base}/v1/messages");
        let client = self.client.clone();
        let api_version = self.api_version;
        let started = Instant::now();
        let request_id = ctx.request_id.clone();

        let resp = with_deadline(ctx.deadline, started, async move {
            client
                .post(&url)
                .header("x-api-key", key)
                .header("anthropic-version", api_version)
                .header(header::CONTENT_TYPE, "application/json")
                .header(header::ACCEPT, "text/event-stream")
                .header("x-aisix-request-id", &request_id)
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

        let byte_stream = resp.bytes_stream();
        let stream = build_chunk_stream(byte_stream);
        Ok(Box::pin(stream))
    }
}

fn build_chunk_stream<S>(
    byte_stream: S,
) -> impl futures::Stream<Item = Result<ChatChunk, BridgeError>> + Send
where
    S: futures::Stream<Item = reqwest::Result<bytes::Bytes>> + Send + 'static,
{
    async_stream::try_stream! {
        let mut decoder = SseDecoder::new();
        let mut stream = Box::pin(byte_stream);
        let mut state = StreamState::default();

        while let Some(next) = stream.next().await {
            let chunk = next.map_err(|e| BridgeError::Transport(e.to_string()))?;
            for event in decoder.feed(chunk.as_ref()) {
                let SseEvent::Data(payload) = event else { continue };
                let parsed: AnthropicStreamEvent = serde_json::from_str(&payload)
                    .map_err(|e| BridgeError::UpstreamDecode(e.to_string()))?;
                state.update(&parsed);
                if let Some(c) = state.to_chunk(&parsed) {
                    yield c;
                }
                if StreamState::is_terminal(&parsed) {
                    return;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aisix_core::{Model, ProviderKey};
    use aisix_gateway::{ChatMessage, FinishReason, Role};
    use std::sync::Arc;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn sample_model() -> Arc<Model> {
        Arc::new(
            serde_json::from_str(
                r#"{
                    "display_name": "my-claude",
                    "provider": "anthropic",
                    "model_name": "claude-sonnet-4-5",
                    "provider_key_id": "11111111-1111-1111-1111-111111111111"
                }"#,
            )
            .unwrap(),
        )
    }

    fn sample_provider_key(base: &str) -> Arc<ProviderKey> {
        let cfg = format!(
            r#"{{"display_name":"anthropic-prod","secret":"sk-ant-test","api_base":"{base}"}}"#
        );
        Arc::new(serde_json::from_str(&cfg).unwrap())
    }

    fn sample_ctx(base: &str) -> BridgeContext {
        BridgeContext::new("req-1", sample_model(), sample_provider_key(base))
    }

    fn req() -> ChatFormat {
        ChatFormat::new(
            "my-claude",
            vec![
                ChatMessage::system("you are helpful"),
                ChatMessage::user("hi"),
            ],
        )
    }

    #[tokio::test]
    async fn non_streaming_happy_path() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(header("x-api-key", "sk-ant-test"))
            .and(header("anthropic-version", "2023-06-01"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "msg_01",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-5",
                "content": [{"type": "text", "text": "hello back"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 2, "output_tokens": 3}
            })))
            .mount(&server)
            .await;

        let bridge = AnthropicBridge::new();
        let ctx = sample_ctx(&server.uri());
        let resp = bridge.chat(&req(), &ctx).await.unwrap();

        assert_eq!(resp.id, "msg_01");
        assert_eq!(resp.message.role, Role::Assistant);
        assert_eq!(resp.message.content, "hello back");
        assert_eq!(resp.finish_reason, FinishReason::Stop);
        assert_eq!(resp.usage.prompt_tokens, 2);
        assert_eq!(resp.usage.completion_tokens, 3);
        assert_eq!(resp.usage.total_tokens, 5);
    }

    #[tokio::test]
    async fn non_streaming_400_bad_request_surfaces_message() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(400).set_body_raw(
                r#"{"error":{"type":"invalid_request","message":"bad"}}"#.as_bytes(),
                "application/json",
            ))
            .mount(&server)
            .await;

        let bridge = AnthropicBridge::new();
        let ctx = sample_ctx(&server.uri());
        let err = bridge.chat(&req(), &ctx).await.unwrap_err();
        match err {
            BridgeError::UpstreamStatus {
                status,
                message,
                parsed,
                ..
            } => {
                assert_eq!(status, 400);
                // After #322: bridge parses Anthropic envelope into a
                // structured view; `message` is now the upstream's
                // `error.message`, not the raw JSON body.
                assert_eq!(message, "bad");
                let parsed = parsed.expect("envelope parsed");
                assert_eq!(parsed.kind.as_deref(), Some("invalid_request"));
                assert_eq!(parsed.message.as_deref(), Some("bad"));
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    /// Issue #543 (audit MEDIUM): the shared `capture_upstream_error_http`
    /// no longer gates parsing on Content-Type, so the Anthropic bridge
    /// must ALSO surface the parsed envelope when the upstream labels a
    /// JSON error body with a non-`application/json` Content-Type. Guards
    /// the shared-fn change on the Anthropic side.
    #[tokio::test]
    async fn non_streaming_400_non_json_content_type_still_surfaces_message() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(400).set_body_raw(
                r#"{"error":{"type":"invalid_request","message":"bad"}}"#.as_bytes(),
                "text/plain",
            ))
            .mount(&server)
            .await;

        let bridge = AnthropicBridge::new();
        let ctx = sample_ctx(&server.uri());
        let err = bridge.chat(&req(), &ctx).await.unwrap_err();
        match err {
            BridgeError::UpstreamStatus {
                status,
                message,
                parsed,
                ..
            } => {
                assert_eq!(status, 400);
                assert_eq!(message, "bad");
                let parsed = parsed.expect("envelope must parse regardless of Content-Type");
                assert_eq!(parsed.kind.as_deref(), Some("invalid_request"));
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[tokio::test]
    async fn non_streaming_decode_error_on_malformed_body() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(200).set_body_string("not-json"))
            .mount(&server)
            .await;

        let bridge = AnthropicBridge::new();
        let ctx = sample_ctx(&server.uri());
        let err = bridge.chat(&req(), &ctx).await.unwrap_err();
        assert!(matches!(err, BridgeError::UpstreamDecode(_)));
    }

    #[tokio::test]
    async fn deadline_elapses_to_timeout_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_delay(Duration::from_secs(5))
                    .set_body_json(serde_json::json!({
                        "id": "x",
                        "type": "message",
                        "role": "assistant",
                        "model": "x",
                        "content": []
                    })),
            )
            .mount(&server)
            .await;

        let bridge = AnthropicBridge::new();
        let ctx = sample_ctx(&server.uri()).with_deadline(Duration::from_millis(50));
        let err = bridge.chat(&req(), &ctx).await.unwrap_err();
        assert!(matches!(err, BridgeError::Timeout { .. }));
    }

    #[tokio::test]
    async fn missing_api_key_is_a_config_error() {
        let mut pk: ProviderKey =
            serde_json::from_str(r#"{"display_name":"empty","secret":"placeholder"}"#).unwrap();
        pk.secret.clear();

        let bridge = AnthropicBridge::new();
        let ctx = BridgeContext::new("req-1", sample_model(), Arc::new(pk));
        let err = bridge.chat(&req(), &ctx).await.unwrap_err();
        assert!(matches!(err, BridgeError::InvalidUpstreamConfig(_)));
    }

    #[tokio::test]
    async fn secret_with_control_chars_is_invalid_config() {
        // A non-empty secret that can't be an x-api-key header value
        // (control bytes) is customer-fixable config, not a 500 (#367).
        let pk: ProviderKey =
            serde_json::from_str(r#"{"display_name":"bad","secret":"sk-live\n-injected"}"#)
                .unwrap();
        let bridge = AnthropicBridge::new();
        let ctx = BridgeContext::new("req-1", sample_model(), Arc::new(pk));
        let err = bridge.chat(&req(), &ctx).await.unwrap_err();
        match err {
            BridgeError::InvalidUpstreamConfig(msg) => {
                assert!(msg.contains("invalid header characters"), "got {msg}");
            }
            other => panic!("expected InvalidUpstreamConfig, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn tool_role_without_tool_call_id_is_rejected_as_config_error() {
        // Tool role IS supported (translates to Anthropic
        // `{role:"user", content:[{type:"tool_result", ...}]}`)
        // when paired with a tool_call_id. Without one, there's no
        // way to pair the result with its originating tool_use, so
        // the gateway rejects with Config.
        let server = MockServer::start().await;
        let bridge = AnthropicBridge::new();
        let ctx = sample_ctx(&server.uri());
        let req = ChatFormat::new(
            "my-claude",
            vec![ChatMessage {
                role: Role::Tool,
                content: "tool output".into(),
                content_blocks: None,
                name: None,
                tool_call_id: None,
                extra: serde_json::Map::new(),
            }],
        );
        let err = bridge.chat(&req, &ctx).await.unwrap_err();
        assert!(matches!(err, BridgeError::InvalidUpstreamConfig(_)));
    }

    #[tokio::test]
    async fn streaming_happy_path_emits_text_deltas_then_finish() {
        let server = MockServer::start().await;
        let sse = "\
event: message_start\n\
data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_stream\",\"model\":\"claude-sonnet-4-5\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"stop_reason\":null,\"usage\":{\"input_tokens\":1}}}\n\n\
event: content_block_start\n\
data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hel\"}}\n\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"lo\"}}\n\n\
event: content_block_stop\n\
data: {\"type\":\"content_block_stop\",\"index\":0}\n\n\
event: message_delta\n\
data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":5}}\n\n\
event: message_stop\n\
data: {\"type\":\"message_stop\"}\n\n";
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse),
            )
            .mount(&server)
            .await;

        let bridge = AnthropicBridge::new();
        let ctx = sample_ctx(&server.uri());
        let mut stream = bridge.chat_stream(&req(), &ctx).await.unwrap();

        let mut chunks = Vec::new();
        while let Some(item) = stream.next().await {
            chunks.push(item.unwrap());
        }
        // Expect: two text deltas, then one message_delta finish chunk.
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].id, "msg_stream");
        assert_eq!(chunks[0].delta.content.as_deref(), Some("hel"));
        assert_eq!(chunks[1].delta.content.as_deref(), Some("lo"));
        assert_eq!(chunks[2].finish_reason, Some(FinishReason::Stop));
        assert_eq!(chunks[2].usage.as_ref().unwrap().completion_tokens, 5);
    }

    #[tokio::test]
    async fn streaming_upstream_error_surfaces_before_stream_start() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(500).set_body_string("oops"))
            .mount(&server)
            .await;

        let bridge = AnthropicBridge::new();
        let ctx = sample_ctx(&server.uri());
        match bridge.chat_stream(&req(), &ctx).await {
            Ok(_) => panic!("expected upstream error"),
            Err(BridgeError::UpstreamStatus { status: 500, .. }) => {}
            Err(other) => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn resolve_base_honours_override() {
        // Default path: ProviderKey has no api_base → falls back to
        // ANTHROPIC_DEFAULT_BASE.
        let pk_default: ProviderKey =
            serde_json::from_str(r#"{"display_name":"x","secret":"k"}"#).unwrap();
        let ctx = BridgeContext::new("rid", sample_model(), Arc::new(pk_default));
        assert!(!resolve_base(&ctx).unwrap().is_empty());

        // api_base override: trailing slash stripped.
        let pk_override: ProviderKey = serde_json::from_str(
            r#"{"display_name":"x","secret":"k","api_base":"https://proxy.example.com/"}"#,
        )
        .unwrap();
        let ctx = BridgeContext::new("rid", sample_model(), Arc::new(pk_override));
        assert_eq!(resolve_base(&ctx).unwrap(), "https://proxy.example.com");
    }

    fn pk_with_base(api_base: &str) -> ProviderKey {
        let cfg = format!(r#"{{"display_name":"x","secret":"k","api_base":"{api_base}"}}"#);
        serde_json::from_str(&cfg).unwrap()
    }

    /// All four Anthropic api_base forms a real operator might paste must
    /// collapse to the canonical bare host. The bridge appends
    /// `/v1/messages` itself at request time.
    #[test]
    fn anthropic_api_base_tolerance_bare_host_v1_and_full_messages_path() {
        let canonical = "https://api.anthropic.com";

        for form in [
            "https://api.anthropic.com",
            "https://api.anthropic.com/",
            "https://api.anthropic.com/v1",
            "https://api.anthropic.com/v1/",
            "https://api.anthropic.com/v1/messages",
            "https://api.anthropic.com/v1/messages/",
            "  https://api.anthropic.com  ",
        ] {
            let pk = pk_with_base(form);
            let ctx = BridgeContext::new("rid", sample_model(), Arc::new(pk));
            assert_eq!(
                resolve_base(&ctx).unwrap(),
                canonical,
                "form {form:?} should normalize to {canonical}",
            );
        }
    }

    /// Same normalization applies to a custom proxy host — operator pastes
    /// whichever form their proxy URL takes, all converge to the bare
    /// host the bridge can append `/v1/messages` to.
    #[test]
    fn anthropic_api_base_tolerance_custom_proxy_host() {
        let canonical = "https://proxy.example.com";

        for form in [
            "https://proxy.example.com",
            "https://proxy.example.com/v1",
            "https://proxy.example.com/v1/messages",
        ] {
            let pk = pk_with_base(form);
            let ctx = BridgeContext::new("rid", sample_model(), Arc::new(pk));
            assert_eq!(resolve_base(&ctx).unwrap(), canonical);
        }
    }

    /// Family-bridge safety: when `AnthropicBridge` serves a non-anthropic
    /// vendor with empty `api_base` it MUST refuse rather than fall back
    /// to `ANTHROPIC_DEFAULT_BASE` — that fallback would silently route
    /// the vendor's API key to `api.anthropic.com`. Mirror of the OpenAI
    /// guard in `crates/aisix-provider-openai/src/bridge.rs`.
    #[test]
    fn family_bridge_refuses_non_anthropic_vendor_with_empty_api_base() {
        for vendor in [
            "bedrock-anthropic",
            "vertex-anthropic",
            "BedrockAnthropic",
            " bedrock-anthropic ",
        ] {
            let pk: ProviderKey = serde_json::from_str(&format!(
                r#"{{"display_name":"x","secret":"k","provider":"{vendor}","adapter":"anthropic"}}"#
            ))
            .unwrap();
            let ctx = BridgeContext::new("rid", sample_model(), Arc::new(pk));
            let err = resolve_base(&ctx).unwrap_err();
            match err {
                BridgeError::InvalidUpstreamConfig(msg) => {
                    assert!(
                        msg.contains("base URL") && msg.contains(vendor.trim()),
                        "vendor {vendor:?}: error must name vendor + base URL; got: {msg}",
                    );
                    // Sensitive-info-leakage guard: internal product
                    // taxonomy must not leak into the customer-visible
                    // 500 body. Those identifiers go to tracing only.
                    for forbidden in ["cp-api", "adapter_map", "provider_metadata"] {
                        assert!(
                            !msg.contains(forbidden),
                            "vendor {vendor:?}: error body must not leak \
                             internal-product taxonomy {forbidden:?}; got: {msg}",
                        );
                    }
                }
                other => {
                    panic!("vendor {vendor:?}: expected InvalidUpstreamConfig, got {other:?}")
                }
            }
        }
    }

    /// Pure-anthropic PK without `api_base` falls back to
    /// `ANTHROPIC_DEFAULT_BASE` — the historical legacy behavior. The
    /// safety check above only fires for non-anthropic vendors.
    #[test]
    fn family_bridge_allows_anthropic_vendor_with_empty_api_base() {
        let pk: ProviderKey = serde_json::from_str(
            r#"{"display_name":"a","secret":"sk-a","provider":"anthropic","adapter":"anthropic"}"#,
        )
        .unwrap();
        let ctx = BridgeContext::new("rid", sample_model(), Arc::new(pk));
        assert_eq!(resolve_base(&ctx).unwrap(), ANTHROPIC_DEFAULT_BASE);
    }

    /// Pre-Phase-A PK carries an empty `provider` string. The safety
    /// check must NOT fire here — those rows route via the compat
    /// shim in `crates/aisix-proxy/src/dispatch.rs::resolve_bridge`
    /// to the specialized "anthropic" bridge. The bridge itself must
    /// tolerate the legacy shape so the compat path doesn't 500.
    #[test]
    fn family_bridge_allows_legacy_empty_provider_with_empty_api_base() {
        let pk: ProviderKey = serde_json::from_str(r#"{"display_name":"x","secret":"k"}"#).unwrap();
        let ctx = BridgeContext::new("rid", sample_model(), Arc::new(pk));
        assert_eq!(resolve_base(&ctx).unwrap(), ANTHROPIC_DEFAULT_BASE);
    }
}
