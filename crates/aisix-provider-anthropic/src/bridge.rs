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
    name: &'static str,
    api_version: &'static str,
}

impl AnthropicBridge {
    pub fn new() -> Self {
        Self::with_client(default_client())
    }

    pub fn with_client(client: Client) -> Self {
        Self {
            client,
            name: "anthropic",
            api_version: ANTHROPIC_VERSION,
        }
    }

    pub fn with_name(mut self, name: &'static str) -> Self {
        self.name = name;
        self
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

fn resolve_base(model: &aisix_core::Model) -> String {
    match model.base_url() {
        Some(b) if !b.trim().is_empty() => b.trim_end_matches('/').to_string(),
        _ => ANTHROPIC_DEFAULT_BASE.to_string(),
    }
}

fn api_key(model: &aisix_core::Model) -> Result<&str, BridgeError> {
    let k = &model.provider_config.api_key;
    if k.is_empty() {
        Err(BridgeError::Config(
            "provider_config.api_key is empty".into(),
        ))
    } else {
        Ok(k.as_str())
    }
}

fn upstream_model(model: &aisix_core::Model) -> Result<&str, BridgeError> {
    model
        .upstream_model()
        .ok_or_else(|| BridgeError::Config("model field missing `provider/` prefix".into()))
}

async fn map_http_error(status: StatusCode, resp: reqwest::Response) -> BridgeError {
    let message = resp.text().await.unwrap_or_default();
    BridgeError::UpstreamStatus {
        status: status.as_u16(),
        message: truncate(&message, 1024),
    }
}

fn truncate(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_string()
    } else {
        format!("{}…", &s[..n])
    }
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
        self.name
    }

    async fn chat(
        &self,
        req: &ChatFormat,
        ctx: &BridgeContext,
    ) -> Result<ChatResponse, BridgeError> {
        let model = ctx.model.as_ref();
        let base = resolve_base(model);
        let key = api_key(model)?;
        let upstream = upstream_model(model)?;

        let (system, messages) =
            split_system(req).map_err(|e| BridgeError::Config(e.to_string()))?;
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
        let model = ctx.model.as_ref();
        let base = resolve_base(model);
        let key = api_key(model)?;
        let upstream = upstream_model(model)?;

        let (system, messages) =
            split_system(req).map_err(|e| BridgeError::Config(e.to_string()))?;
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
    use aisix_core::Model;
    use aisix_gateway::{ChatMessage, FinishReason, Role};
    use std::sync::Arc;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn sample_model(base: &str) -> Arc<Model> {
        let cfg = format!(
            r#"{{
                "name": "my-claude",
                "model": "anthropic/claude-sonnet-4-5",
                "provider_config": {{"api_key": "sk-ant-test", "api_base": "{base}"}}
            }}"#
        );
        Arc::new(serde_json::from_str(&cfg).unwrap())
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
        let ctx = BridgeContext::new("req-1", sample_model(&server.uri()));
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
            .respond_with(
                ResponseTemplate::new(400)
                    .set_body_string(r#"{"error":{"type":"invalid_request","message":"bad"}}"#),
            )
            .mount(&server)
            .await;

        let bridge = AnthropicBridge::new();
        let ctx = BridgeContext::new("req-1", sample_model(&server.uri()));
        let err = bridge.chat(&req(), &ctx).await.unwrap_err();
        match err {
            BridgeError::UpstreamStatus { status, message } => {
                assert_eq!(status, 400);
                assert!(message.contains("invalid_request"));
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
        let ctx = BridgeContext::new("req-1", sample_model(&server.uri()));
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
        let ctx = BridgeContext::new("req-1", sample_model(&server.uri()))
            .with_deadline(Duration::from_millis(50));
        let err = bridge.chat(&req(), &ctx).await.unwrap_err();
        assert!(matches!(err, BridgeError::Timeout { .. }));
    }

    #[tokio::test]
    async fn missing_api_key_is_a_config_error() {
        let mut model: Model = serde_json::from_str(
            r#"{
                "name": "bad",
                "model": "anthropic/claude-sonnet-4-5",
                "provider_config": {"api_key": "placeholder"}
            }"#,
        )
        .unwrap();
        model.provider_config.api_key.clear();

        let bridge = AnthropicBridge::new();
        let ctx = BridgeContext::new("req-1", Arc::new(model));
        let err = bridge.chat(&req(), &ctx).await.unwrap_err();
        assert!(matches!(err, BridgeError::Config(_)));
    }

    #[tokio::test]
    async fn tool_role_is_rejected_as_config_error() {
        let server = MockServer::start().await;
        let bridge = AnthropicBridge::new();
        let ctx = BridgeContext::new("req-1", sample_model(&server.uri()));
        let req = ChatFormat::new(
            "my-claude",
            vec![ChatMessage {
                role: Role::Tool,
                content: "tool output".into(),
                name: None,
                tool_call_id: Some("tc_1".into()),
            }],
        );
        let err = bridge.chat(&req, &ctx).await.unwrap_err();
        assert!(matches!(err, BridgeError::Config(_)));
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
        let ctx = BridgeContext::new("req-1", sample_model(&server.uri()));
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
        let ctx = BridgeContext::new("req-1", sample_model(&server.uri()));
        match bridge.chat_stream(&req(), &ctx).await {
            Ok(_) => panic!("expected upstream error"),
            Err(BridgeError::UpstreamStatus { status: 500, .. }) => {}
            Err(other) => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn resolve_base_honours_override() {
        let mut m: Model = serde_json::from_str(
            r#"{
                "name": "x",
                "model": "anthropic/claude-sonnet-4-5",
                "provider_config": {"api_key": "k"}
            }"#,
        )
        .unwrap();
        // Default path: Provider::Anthropic.default_base_url() comes
        // from aisix-core, so just assert that *some* non-empty host
        // is picked up.
        assert!(!resolve_base(&m).is_empty());

        m.provider_config.api_base = Some("https://proxy.example.com/".into());
        assert_eq!(resolve_base(&m), "https://proxy.example.com");
    }
}
