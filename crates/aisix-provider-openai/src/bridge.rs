//! `OpenAiBridge` — concrete [`Bridge`] implementation for OpenAI.
//!
//! Also reusable by OpenAI-compatible providers (DeepSeek, Gemini's
//! OpenAI-compat endpoint) since their request/response wire shapes
//! match the `/chat/completions` surface almost exactly. Those crates
//! can wrap this bridge or construct it with a different `api_base`.
//!
//! Transport layer:
//! - `reqwest::Client` is shared across requests (connection reuse).
//! - `Authorization: Bearer <api_key>` sourced from the
//!   [`aisix_core::Model`]'s `provider_config`.
//! - Timeout comes from `BridgeContext::deadline` when present;
//!   otherwise the request runs to completion.
//!
//! Error mapping:
//! - reqwest transport error → `BridgeError::Transport`
//! - upstream non-2xx → `BridgeError::UpstreamStatus` (4xx passes through,
//!   5xx collapses to 502 via `http_status()` on the proxy side).
//! - malformed JSON from upstream → `BridgeError::UpstreamDecode`
//! - elapsed deadline → `BridgeError::Timeout { elapsed_ms }`

use aisix_gateway::{
    Bridge, BridgeContext, BridgeError, ChatChunk, ChatChunkStream, ChatFormat, ChatResponse,
    EmbeddingRequest, EmbeddingResponse, SseDecoder, SseEvent,
};
use async_trait::async_trait;
use futures::StreamExt;
use reqwest::{header, Client, StatusCode};
use std::time::{Duration, Instant};

use crate::wire::{
    build_request, embed_request_body, embed_response_into, messages_from,
    response_into_chat_response, stream_chunk_into_chat_chunk, OpenAiEmbedResponse, OpenAiResponse,
    OpenAiStreamChunk,
};

/// Fallback OpenAI host used when the Model doesn't set `api_base` and
/// the Provider enum's default is also missing. In practice an operator
/// configures `api_base: "https://api.openai.com/v1"` on the Model so
/// this constant only covers degenerate config paths.
pub const OPENAI_DEFAULT_BASE: &str = "https://api.openai.com/v1";

pub struct OpenAiBridge {
    client: Client,
    name: &'static str,
}

impl OpenAiBridge {
    pub fn new() -> Self {
        Self::with_client(default_client())
    }

    pub fn with_client(client: Client) -> Self {
        Self {
            client,
            name: "openai",
        }
    }

    /// Same transport but a caller-chosen `name()` — used by OpenAI-compat
    /// providers (DeepSeek, Gemini-OAI) so their metrics labels are distinct.
    pub fn with_name(mut self, name: &'static str) -> Self {
        self.name = name;
        self
    }
}

impl Default for OpenAiBridge {
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
        _ => OPENAI_DEFAULT_BASE.to_string(),
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

#[async_trait]
impl Bridge for OpenAiBridge {
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

        let messages = messages_from(req);
        let body = build_request(req, upstream, &messages, false);
        let url = format!("{base}/chat/completions");
        let client = self.client.clone();
        let started = Instant::now();
        let request_id = ctx.request_id.clone();

        with_deadline(ctx.deadline, started, async move {
            let resp = client
                .post(&url)
                .header(header::AUTHORIZATION, format!("Bearer {key}"))
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

            let parsed: OpenAiResponse = resp
                .json()
                .await
                .map_err(|e| BridgeError::UpstreamDecode(e.to_string()))?;
            Ok(response_into_chat_response(parsed))
        })
        .await
    }

    async fn embed(
        &self,
        req: &EmbeddingRequest,
        ctx: &BridgeContext,
    ) -> Result<EmbeddingResponse, BridgeError> {
        let model = ctx.model.as_ref();
        let base = resolve_base(model);
        let key = api_key(model)?;
        let upstream = upstream_model(model)?;

        let body = embed_request_body(req, upstream);
        let url = format!("{base}/embeddings");
        let client = self.client.clone();
        let started = Instant::now();
        let request_id = ctx.request_id.clone();

        with_deadline(ctx.deadline, started, async move {
            let resp = client
                .post(&url)
                .header(header::AUTHORIZATION, format!("Bearer {key}"))
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

            let parsed: OpenAiEmbedResponse = resp
                .json()
                .await
                .map_err(|e| BridgeError::UpstreamDecode(e.to_string()))?;
            Ok(embed_response_into(parsed))
        })
        .await
    }

    async fn complete(
        &self,
        body: &serde_json::Value,
        ctx: &BridgeContext,
    ) -> Result<serde_json::Value, BridgeError> {
        let model = ctx.model.as_ref();
        let base = resolve_base(model);
        let key = api_key(model)?;
        let upstream = upstream_model(model)?;

        // Replace the `model` field with the upstream provider id.
        let mut outbound = body.clone();
        if let Some(obj) = outbound.as_object_mut() {
            obj.insert("model".to_string(), serde_json::Value::String(upstream.to_string()));
        }

        let url = format!("{base}/completions");
        let client = self.client.clone();
        let started = Instant::now();
        let request_id = ctx.request_id.clone();

        with_deadline(ctx.deadline, started, async move {
            let resp = client
                .post(&url)
                .header(header::AUTHORIZATION, format!("Bearer {key}"))
                .header(header::CONTENT_TYPE, "application/json")
                .header("x-aisix-request-id", &request_id)
                .json(&outbound)
                .send()
                .await
                .map_err(|e| BridgeError::Transport(e.to_string()))?;

            let status = resp.status();
            if !status.is_success() {
                return Err(map_http_error(status, resp).await);
            }

            resp.json::<serde_json::Value>()
                .await
                .map_err(|e| BridgeError::UpstreamDecode(e.to_string()))
        })
        .await
    }

    async fn generate_image(
        &self,
        body: &serde_json::Value,
        ctx: &BridgeContext,
    ) -> Result<serde_json::Value, BridgeError> {
        let model = ctx.model.as_ref();
        let base = resolve_base(model);
        let key = api_key(model)?;
        let upstream = upstream_model(model)?;

        // Replace the `model` field with the upstream provider id.
        let mut outbound = body.clone();
        if let Some(obj) = outbound.as_object_mut() {
            obj.insert(
                "model".to_string(),
                serde_json::Value::String(upstream.to_string()),
            );
        }

        let url = format!("{base}/images/generations");
        let client = self.client.clone();
        let started = Instant::now();
        let request_id = ctx.request_id.clone();

        with_deadline(ctx.deadline, started, async move {
            let resp = client
                .post(&url)
                .header(header::AUTHORIZATION, format!("Bearer {key}"))
                .header(header::CONTENT_TYPE, "application/json")
                .header("x-aisix-request-id", &request_id)
                .json(&outbound)
                .send()
                .await
                .map_err(|e| BridgeError::Transport(e.to_string()))?;

            let status = resp.status();
            if !status.is_success() {
                return Err(map_http_error(status, resp).await);
            }

            resp.json::<serde_json::Value>()
                .await
                .map_err(|e| BridgeError::UpstreamDecode(e.to_string()))
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

        let messages = messages_from(req);
        let body = build_request(req, upstream, &messages, true);
        let url = format!("{base}/chat/completions");
        let client = self.client.clone();
        let started = Instant::now();
        let request_id = ctx.request_id.clone();

        let resp = with_deadline(ctx.deadline, started, async move {
            client
                .post(&url)
                .header(header::AUTHORIZATION, format!("Bearer {key}"))
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
        while let Some(next) = stream.next().await {
            let chunk = next.map_err(|e| BridgeError::Transport(e.to_string()))?;
            for event in decoder.feed(chunk.as_ref()) {
                match event {
                    SseEvent::Done => return,
                    SseEvent::Data(payload) => {
                        let parsed: OpenAiStreamChunk = serde_json::from_str(&payload)
                            .map_err(|e| BridgeError::UpstreamDecode(e.to_string()))?;
                        yield stream_chunk_into_chat_chunk(parsed);
                    }
                }
            }
        }
        if let Some(SseEvent::Data(payload)) = decoder.finish() {
            let parsed: OpenAiStreamChunk = serde_json::from_str(&payload)
                .map_err(|e| BridgeError::UpstreamDecode(e.to_string()))?;
            yield stream_chunk_into_chat_chunk(parsed);
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
                "name": "my-gpt4",
                "model": "openai/gpt-4o",
                "provider_config": {{"api_key": "sk-test", "api_base": "{base}"}}
            }}"#
        );
        Arc::new(serde_json::from_str(&cfg).unwrap())
    }

    fn req() -> ChatFormat {
        ChatFormat::new("my-gpt4", vec![ChatMessage::user("hi")])
    }

    #[tokio::test]
    async fn non_streaming_happy_path() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(header("authorization", "Bearer sk-test"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-1",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "hello back"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4}
            })))
            .mount(&server)
            .await;

        let bridge = OpenAiBridge::new();
        let ctx = BridgeContext::new("req-1", sample_model(&server.uri()));
        let resp = bridge.chat(&req(), &ctx).await.unwrap();

        assert_eq!(resp.id, "cmpl-1");
        assert_eq!(resp.message.role, Role::Assistant);
        assert_eq!(resp.message.content, "hello back");
        assert_eq!(resp.finish_reason, FinishReason::Stop);
        assert_eq!(resp.usage.total_tokens, 4);
    }

    #[tokio::test]
    async fn non_streaming_429_maps_to_upstream_status() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(429).set_body_string("slow down"))
            .mount(&server)
            .await;

        let bridge = OpenAiBridge::new();
        let ctx = BridgeContext::new("req-1", sample_model(&server.uri()));
        let err = bridge.chat(&req(), &ctx).await.unwrap_err();
        match err {
            BridgeError::UpstreamStatus { status, message } => {
                assert_eq!(status, 429);
                assert!(message.contains("slow down"));
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[tokio::test]
    async fn non_streaming_decode_error_on_malformed_body() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_string("not-json"))
            .mount(&server)
            .await;

        let bridge = OpenAiBridge::new();
        let ctx = BridgeContext::new("req-1", sample_model(&server.uri()));
        let err = bridge.chat(&req(), &ctx).await.unwrap_err();
        assert!(matches!(err, BridgeError::UpstreamDecode(_)));
    }

    #[tokio::test]
    async fn deadline_elapses_to_timeout_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_delay(Duration::from_secs(5))
                    .set_body_json(serde_json::json!({"id":"x","model":"x","choices":[]})),
            )
            .mount(&server)
            .await;

        let bridge = OpenAiBridge::new();
        let ctx = BridgeContext::new("req-1", sample_model(&server.uri()))
            .with_deadline(Duration::from_millis(50));
        let err = bridge.chat(&req(), &ctx).await.unwrap_err();
        assert!(matches!(err, BridgeError::Timeout { .. }));
    }

    #[tokio::test]
    async fn missing_api_key_is_a_config_error() {
        // Construct a Model whose api_key is empty. We bypass JSON Schema
        // here because the loader would normally reject this, but tests
        // of the Bridge's own guard still need the path exercised.
        let mut model: Model = serde_json::from_str(
            r#"{
                "name": "bad",
                "model": "openai/gpt-4o",
                "provider_config": {"api_key": "placeholder"}
            }"#,
        )
        .unwrap();
        model.provider_config.api_key.clear();

        let bridge = OpenAiBridge::new();
        let ctx = BridgeContext::new("req-1", Arc::new(model));
        let err = bridge.chat(&req(), &ctx).await.unwrap_err();
        assert!(matches!(err, BridgeError::Config(_)));
    }

    #[tokio::test]
    async fn streaming_happy_path_emits_chunks_then_done() {
        let server = MockServer::start().await;
        let sse = "\
data: {\"id\":\"cmpl-s\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"cmpl-s\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hel\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"cmpl-s\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"},\"finish_reason\":\"stop\"}]}\n\n\
data: [DONE]\n\n";
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse),
            )
            .mount(&server)
            .await;

        let bridge = OpenAiBridge::new();
        let ctx = BridgeContext::new("req-1", sample_model(&server.uri()));
        let mut stream = bridge.chat_stream(&req(), &ctx).await.unwrap();

        let mut chunks = Vec::new();
        while let Some(item) = stream.next().await {
            chunks.push(item.unwrap());
        }
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].delta.role, Some(Role::Assistant));
        assert_eq!(chunks[1].delta.content.as_deref(), Some("hel"));
        assert_eq!(chunks[2].delta.content.as_deref(), Some("lo"));
        assert_eq!(chunks[2].finish_reason, Some(FinishReason::Stop));
    }

    #[tokio::test]
    async fn streaming_upstream_error_surfaces_before_stream_start() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(500).set_body_string("oops"))
            .mount(&server)
            .await;

        let bridge = OpenAiBridge::new();
        let ctx = BridgeContext::new("req-1", sample_model(&server.uri()));
        match bridge.chat_stream(&req(), &ctx).await {
            Ok(_) => panic!("expected upstream error, got a live stream"),
            Err(BridgeError::UpstreamStatus { status: 500, .. }) => {}
            Err(other) => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn resolve_base_trims_trailing_slash_and_honours_override() {
        let mut m: Model = serde_json::from_str(
            r#"{
                "name": "x",
                "model": "openai/gpt-4o",
                "provider_config": {"api_key": "k"}
            }"#,
        )
        .unwrap();
        // No api_base set → Provider::Openai's default host.
        assert_eq!(resolve_base(&m), "https://api.openai.com");

        m.provider_config.api_base = Some("https://proxy.example.com/v1/".into());
        assert_eq!(resolve_base(&m), "https://proxy.example.com/v1");
    }
}
