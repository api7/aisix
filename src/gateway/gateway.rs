use std::{pin::Pin, time::Duration};

use futures::Stream;
use http::StatusCode;
use serde_json::Value;
use tokio::sync::oneshot;

use crate::gateway::{
    error::{GatewayError, Result},
    formats::OpenAIChatFormat,
    provider_instance::{ProviderInstance, ProviderRegistry},
    streams::{BridgedStream, HubChunkStream, NativeStream, sse_reader},
    traits::{ChatFormat, NativeHandler, StreamReaderKind},
    types::{
        common::Usage,
        openai::{ChatCompletionRequest, ChatCompletionResponse},
        response::ChatResponse,
    },
};

/// Typed Layer-3 gateway entry point.
pub struct Gateway {
    registry: ProviderRegistry,
    http_client: reqwest::Client,
}

impl Gateway {
    const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
    const COMPLETE_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

    /// Creates a new gateway with the provided provider registry.
    pub fn new(registry: ProviderRegistry) -> Self {
        let http_client = reqwest::Client::builder()
            .connect_timeout(Self::CONNECT_TIMEOUT)
            .build()
            .expect("failed to build gateway reqwest client with configured timeouts");

        Self {
            registry,
            http_client,
        }
    }

    /// Returns the immutable provider registry backing this gateway.
    pub fn registry(&self) -> &ProviderRegistry {
        &self.registry
    }

    /// Typed chat entry point for both complete and streaming requests.
    #[fastrace::trace]
    pub async fn chat<F: ChatFormat>(
        &self,
        request: &F::Request,
        instance: &ProviderInstance,
    ) -> Result<ChatResponse<F>> {
        let stream = F::is_stream(request);

        if let Some(native) = F::native_support(instance.def.as_ref()) {
            return self
                .call_chat_native::<F>(&native, instance, request, stream)
                .await;
        }

        let (hub_request, ctx) = F::to_hub(request)?;

        if stream {
            let hub_stream = self.call_chat_hub_stream(instance, &hub_request).await?;
            let (usage_tx, usage_rx) = oneshot::channel();
            let bridged_stream = BridgedStream::<F>::new(hub_stream, ctx, usage_tx);

            return Ok(ChatResponse::Stream {
                stream: Box::pin(bridged_stream),
                usage_rx,
            });
        }

        let hub_response = self.call_chat_hub(instance, &hub_request).await?;
        let usage = extract_chat_usage_from_response(&hub_response).unwrap_or_default();
        let response = F::from_hub(&hub_response, &ctx)?;

        Ok(ChatResponse::Complete { response, usage })
    }

    /// Convenience wrapper for the OpenAI Chat format.
    #[fastrace::trace]
    pub async fn chat_completion(
        &self,
        request: &ChatCompletionRequest,
        instance: &ProviderInstance,
    ) -> Result<ChatResponse<OpenAIChatFormat>> {
        self.chat::<OpenAIChatFormat>(request, instance).await
    }

    async fn send_chat_request(
        &self,
        request: reqwest::RequestBuilder,
        stream: bool,
    ) -> Result<reqwest::Response> {
        let request = if stream {
            request
        } else {
            request.timeout(Self::COMPLETE_REQUEST_TIMEOUT)
        };

        request.send().await.map_err(GatewayError::Http)
    }

    async fn call_chat_native<F: ChatFormat>(
        &self,
        native: &NativeHandler<'_>,
        instance: &ProviderInstance,
        request: &F::Request,
        stream: bool,
    ) -> Result<ChatResponse<F>> {
        let (endpoint_path, body) = F::call_native(native, request, stream)?;
        if stream {
            ensure_chat_stream_reader_supported(instance.def.stream_reader_kind())?;
        }

        let base_url = instance.effective_base_url()?;
        let url = join_url(base_url.as_str(), &endpoint_path);
        let headers = instance.build_headers()?;

        let request = self.http_client.post(url).headers(headers).json(&body);
        let response = self.send_chat_request(request, stream).await?;

        if !response.status().is_success() {
            return Err(provider_error(response, instance.def.name()).await);
        }

        if stream {
            let raw_chunks =
                select_chat_stream_reader(instance.def.stream_reader_kind(), response)?;
            let (usage_tx, usage_rx) = oneshot::channel();
            let native_stream = NativeStream::<F>::new(raw_chunks, instance.def.clone(), usage_tx);

            return Ok(ChatResponse::Stream {
                stream: Box::pin(native_stream),
                usage_rx,
            });
        }

        let body: Value = response.json().await.map_err(GatewayError::Http)?;
        let response = F::parse_native_response(native, body)?;

        Ok(ChatResponse::Complete {
            response,
            usage: Usage::default(),
        })
    }

    async fn call_chat_hub(
        &self,
        instance: &ProviderInstance,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse> {
        let provider_body = instance.def.transform_request(request)?;
        let url = instance.build_url(&request.model)?;
        let headers = instance.build_headers()?;

        let request = self
            .http_client
            .post(url)
            .headers(headers)
            .json(&provider_body);
        let response = self.send_chat_request(request, false).await?;

        if !response.status().is_success() {
            return Err(provider_error(response, instance.def.name()).await);
        }

        let body: Value = response.json().await.map_err(GatewayError::Http)?;
        instance.def.transform_response(body)
    }

    async fn call_chat_hub_stream(
        &self,
        instance: &ProviderInstance,
        request: &ChatCompletionRequest,
    ) -> Result<HubChunkStream> {
        ensure_chat_stream_reader_supported(instance.def.stream_reader_kind())?;

        let provider_body = instance.def.transform_request(request)?;
        let url = instance.build_url(&request.model)?;
        let headers = instance.build_headers()?;

        let request = self
            .http_client
            .post(url)
            .headers(headers)
            .json(&provider_body);
        let response = self.send_chat_request(request, true).await?;

        if !response.status().is_success() {
            return Err(provider_error(response, instance.def.name()).await);
        }

        let raw_chunks = select_chat_stream_reader(instance.def.stream_reader_kind(), response)?;
        Ok(HubChunkStream::new(raw_chunks, instance.def.clone()))
    }
}

fn ensure_chat_stream_reader_supported(kind: StreamReaderKind) -> Result<()> {
    match kind {
        StreamReaderKind::Sse => Ok(()),
        other => Err(GatewayError::Validation(format!(
            "stream reader kind {:?} is not implemented yet",
            other
        ))),
    }
}

fn select_chat_stream_reader(
    kind: StreamReaderKind,
    response: reqwest::Response,
) -> Result<Pin<Box<dyn Stream<Item = Result<String>> + Send>>> {
    ensure_chat_stream_reader_supported(kind)?;

    match kind {
        StreamReaderKind::Sse => Ok(sse_reader(response.bytes_stream())),
        StreamReaderKind::AwsEventStream | StreamReaderKind::JsonArrayStream => {
            unreachable!(
                "unsupported stream reader kind should be rejected before response wrapping"
            )
        }
    }
}

fn join_url(base_url: &str, endpoint_path: &str) -> String {
    let base_url = base_url.trim_end_matches('/');
    if endpoint_path.starts_with('/') {
        format!("{base_url}{endpoint_path}")
    } else {
        format!("{base_url}/{endpoint_path}")
    }
}

fn extract_chat_usage_from_response(response: &ChatCompletionResponse) -> Option<Usage> {
    response.usage.as_ref().map(|usage| Usage {
        input_tokens: Some(usage.prompt_tokens),
        output_tokens: Some(usage.completion_tokens),
        total_tokens: Some(usage.total_tokens),
        input_audio_tokens: usage
            .prompt_tokens_details
            .as_ref()
            .and_then(|details| details.audio_tokens),
        output_audio_tokens: usage
            .completion_tokens_details
            .as_ref()
            .and_then(|details| details.audio_tokens),
        cache_read_input_tokens: usage
            .prompt_tokens_details
            .as_ref()
            .and_then(|details| details.cached_tokens),
        ..Default::default()
    })
}

async fn provider_error(response: reqwest::Response, provider: &str) -> GatewayError {
    let status = response.status();
    let body = response
        .bytes()
        .await
        .map(|bytes| {
            serde_json::from_slice(&bytes)
                .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned()))
        })
        .unwrap_or(Value::Null);

    GatewayError::Provider {
        status,
        body,
        provider: provider.to_string(),
        retryable: status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error(),
    }
}

#[cfg(test)]
mod tests {
    use std::{
        borrow::Cow,
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
    };

    use axum::{Json, Router, routing::post};
    use futures::StreamExt;
    use http::{
        HeaderMap, HeaderValue, StatusCode,
        header::{AUTHORIZATION, CONTENT_TYPE, HeaderName},
    };
    use reqwest::Url;
    use serde_json::{Value, json};
    use tokio::{net::TcpListener, sync::Mutex, task::JoinHandle};

    use super::Gateway;
    use crate::gateway::{
        error::{GatewayError, Result},
        provider_instance::{ProviderAuth, ProviderInstance, ProviderRegistry},
        traits::{
            ChatFormat, ChatTransform, NativeHandler, NativeOpenAIResponsesSupport,
            OpenAIResponsesNativeStreamState, ProviderCapabilities, ProviderMeta, StreamReaderKind,
        },
        types::{
            common::{BridgeContext, Usage},
            openai::{
                ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse,
                responses::{ResponsesApiRequest, ResponsesApiResponse, ResponsesApiStreamEvent},
            },
            response::ChatResponse,
        },
    };

    type ObservedRequest = Option<(Option<String>, Value)>;

    struct HubTestProvider;

    struct NativeTestProvider;

    struct UnsupportedHubStreamTestProvider;

    struct UnsupportedNativeStreamTestProvider;

    struct DummyNativeFormat;

    #[derive(Default)]
    struct StreamingNativeState {
        usage: Usage,
    }

    struct StreamingNativeFormat;

    impl ProviderMeta for HubTestProvider {
        fn name(&self) -> &'static str {
            "hub-test"
        }

        fn default_base_url(&self) -> &'static str {
            "https://example.invalid"
        }

        fn chat_endpoint_path(&self, _model: &str) -> Cow<'static, str> {
            Cow::Borrowed("/v1/chat/completions")
        }

        fn stream_reader_kind(&self) -> StreamReaderKind {
            StreamReaderKind::Sse
        }

        fn build_auth_headers(&self, auth: &ProviderAuth) -> Result<HeaderMap> {
            bearer_headers(self.name(), auth)
        }
    }

    impl ChatTransform for HubTestProvider {}

    impl ProviderCapabilities for HubTestProvider {}

    impl ProviderMeta for NativeTestProvider {
        fn name(&self) -> &'static str {
            "native-test"
        }

        fn default_base_url(&self) -> &'static str {
            "https://example.invalid"
        }

        fn stream_reader_kind(&self) -> StreamReaderKind {
            StreamReaderKind::Sse
        }

        fn build_auth_headers(&self, auth: &ProviderAuth) -> Result<HeaderMap> {
            bearer_headers(self.name(), auth)
        }
    }

    impl ChatTransform for NativeTestProvider {}

    impl NativeOpenAIResponsesSupport for NativeTestProvider {
        fn native_openai_responses_endpoint(&self, _model: &str) -> Cow<'static, str> {
            Cow::Borrowed("/v1/native")
        }

        fn transform_openai_responses_request(&self, _req: &ResponsesApiRequest) -> Result<Value> {
            Ok(json!({}))
        }

        fn transform_openai_responses_response(
            &self,
            _body: Value,
        ) -> Result<ResponsesApiResponse> {
            unreachable!("not used in this test")
        }

        fn transform_openai_responses_stream_chunk(
            &self,
            _raw: &str,
            _state: &mut OpenAIResponsesNativeStreamState,
        ) -> Result<Vec<ResponsesApiStreamEvent>> {
            Ok(vec![])
        }
    }

    impl ProviderCapabilities for NativeTestProvider {
        fn as_native_openai_responses(&self) -> Option<&dyn NativeOpenAIResponsesSupport> {
            Some(self)
        }
    }

    impl ProviderMeta for UnsupportedHubStreamTestProvider {
        fn name(&self) -> &'static str {
            "unsupported-hub-stream-test"
        }

        fn default_base_url(&self) -> &'static str {
            "https://example.invalid"
        }

        fn chat_endpoint_path(&self, _model: &str) -> Cow<'static, str> {
            Cow::Borrowed("/v1/chat/completions")
        }

        fn stream_reader_kind(&self) -> StreamReaderKind {
            StreamReaderKind::JsonArrayStream
        }

        fn build_auth_headers(&self, auth: &ProviderAuth) -> Result<HeaderMap> {
            bearer_headers(self.name(), auth)
        }
    }

    impl ChatTransform for UnsupportedHubStreamTestProvider {}

    impl ProviderCapabilities for UnsupportedHubStreamTestProvider {}

    impl ProviderMeta for UnsupportedNativeStreamTestProvider {
        fn name(&self) -> &'static str {
            "unsupported-native-stream-test"
        }

        fn default_base_url(&self) -> &'static str {
            "https://example.invalid"
        }

        fn stream_reader_kind(&self) -> StreamReaderKind {
            StreamReaderKind::AwsEventStream
        }

        fn build_auth_headers(&self, auth: &ProviderAuth) -> Result<HeaderMap> {
            bearer_headers(self.name(), auth)
        }
    }

    impl ChatTransform for UnsupportedNativeStreamTestProvider {}

    impl NativeOpenAIResponsesSupport for UnsupportedNativeStreamTestProvider {
        fn native_openai_responses_endpoint(&self, _model: &str) -> Cow<'static, str> {
            Cow::Borrowed("/v1/native")
        }

        fn transform_openai_responses_request(&self, _req: &ResponsesApiRequest) -> Result<Value> {
            Ok(json!({}))
        }

        fn transform_openai_responses_response(
            &self,
            _body: Value,
        ) -> Result<ResponsesApiResponse> {
            unreachable!("not used in this test")
        }

        fn transform_openai_responses_stream_chunk(
            &self,
            _raw: &str,
            _state: &mut OpenAIResponsesNativeStreamState,
        ) -> Result<Vec<ResponsesApiStreamEvent>> {
            Ok(vec![])
        }
    }

    impl ProviderCapabilities for UnsupportedNativeStreamTestProvider {
        fn as_native_openai_responses(&self) -> Option<&dyn NativeOpenAIResponsesSupport> {
            Some(self)
        }
    }

    impl ChatFormat for DummyNativeFormat {
        type Request = Value;
        type Response = Value;
        type StreamChunk = Value;
        type BridgeState = ();
        type NativeStreamState = ();

        fn name() -> &'static str {
            "dummy_native"
        }

        fn is_stream(_req: &Self::Request) -> bool {
            false
        }

        fn extract_model(req: &Self::Request) -> &str {
            req.get("model")
                .and_then(Value::as_str)
                .unwrap_or("dummy-native-model")
        }

        fn to_hub(_req: &Self::Request) -> Result<(ChatCompletionRequest, BridgeContext)> {
            unreachable!("not used in this test")
        }

        fn from_hub(
            _resp: &ChatCompletionResponse,
            _ctx: &BridgeContext,
        ) -> Result<Self::Response> {
            unreachable!("not used in this test")
        }

        fn from_hub_stream(
            _chunk: &ChatCompletionChunk,
            _state: &mut Self::BridgeState,
            _ctx: &BridgeContext,
        ) -> Result<Vec<Self::StreamChunk>> {
            unreachable!("not used in this test")
        }

        fn native_support(provider: &dyn ProviderCapabilities) -> Option<NativeHandler<'_>>
        where
            Self: Sized,
        {
            provider
                .as_native_openai_responses()
                .map(NativeHandler::OpenAIResponses)
        }

        fn call_native(
            _native: &NativeHandler<'_>,
            request: &Self::Request,
            _stream: bool,
        ) -> Result<(String, Value)>
        where
            Self: Sized,
        {
            Ok(("/v1/native".into(), request.clone()))
        }

        fn transform_native_stream_chunk(
            _provider: &dyn ProviderCapabilities,
            _raw: &str,
            _state: &mut Self::NativeStreamState,
        ) -> Result<Vec<Self::StreamChunk>> {
            Ok(vec![])
        }

        fn parse_native_response(_native: &NativeHandler<'_>, body: Value) -> Result<Self::Response>
        where
            Self: Sized,
        {
            Ok(body)
        }

        fn serialize_chunk_payload(chunk: &Self::StreamChunk) -> String {
            serde_json::to_string(chunk).unwrap()
        }
    }

    impl ChatFormat for StreamingNativeFormat {
        type Request = Value;
        type Response = Value;
        type StreamChunk = Value;
        type BridgeState = ();
        type NativeStreamState = StreamingNativeState;

        fn name() -> &'static str {
            "streaming_native"
        }

        fn is_stream(req: &Self::Request) -> bool {
            req.get("stream").and_then(Value::as_bool).unwrap_or(false)
        }

        fn extract_model(req: &Self::Request) -> &str {
            req.get("model")
                .and_then(Value::as_str)
                .unwrap_or("streaming-native-model")
        }

        fn to_hub(_req: &Self::Request) -> Result<(ChatCompletionRequest, BridgeContext)> {
            unreachable!("not used in this test")
        }

        fn from_hub(
            _resp: &ChatCompletionResponse,
            _ctx: &BridgeContext,
        ) -> Result<Self::Response> {
            unreachable!("not used in this test")
        }

        fn from_hub_stream(
            _chunk: &ChatCompletionChunk,
            _state: &mut Self::BridgeState,
            _ctx: &BridgeContext,
        ) -> Result<Vec<Self::StreamChunk>> {
            unreachable!("not used in this test")
        }

        fn native_support(provider: &dyn ProviderCapabilities) -> Option<NativeHandler<'_>>
        where
            Self: Sized,
        {
            provider
                .as_native_openai_responses()
                .map(NativeHandler::OpenAIResponses)
        }

        fn call_native(
            _native: &NativeHandler<'_>,
            request: &Self::Request,
            stream: bool,
        ) -> Result<(String, Value)>
        where
            Self: Sized,
        {
            let path = if stream {
                "/v1/native-stream"
            } else {
                "/v1/native"
            };
            Ok((path.into(), request.clone()))
        }

        fn transform_native_stream_chunk(
            _provider: &dyn ProviderCapabilities,
            raw: &str,
            state: &mut Self::NativeStreamState,
        ) -> Result<Vec<Self::StreamChunk>> {
            match raw {
                "data: buffered" => Ok(vec![json!({"value": "first"}), json!({"value": "second"})]),
                "data: usage" => {
                    state.usage = Usage {
                        input_tokens: Some(5),
                        output_tokens: Some(8),
                        total_tokens: Some(13),
                        ..Default::default()
                    };
                    Ok(vec![])
                }
                _ => Ok(vec![]),
            }
        }

        fn parse_native_response(_native: &NativeHandler<'_>, body: Value) -> Result<Self::Response>
        where
            Self: Sized,
        {
            Ok(body)
        }

        fn native_usage(state: &Self::NativeStreamState) -> Usage {
            state.usage.clone()
        }

        fn serialize_chunk_payload(chunk: &Self::StreamChunk) -> String {
            serde_json::to_string(chunk).unwrap()
        }
    }

    #[tokio::test]
    async fn chat_completion_uses_hub_path_and_extracts_usage() {
        let observed: Arc<Mutex<ObservedRequest>> = Arc::new(Mutex::new(None));
        let observed_clone = Arc::clone(&observed);
        let router = Router::new().route(
            "/v1/chat/completions",
            post(move |headers: HeaderMap, Json(body): Json<Value>| {
                let observed = Arc::clone(&observed_clone);
                async move {
                    let auth = headers
                        .get(AUTHORIZATION)
                        .and_then(|value| value.to_str().ok())
                        .map(str::to_owned);
                    *observed.lock().await = Some((auth, body));

                    Json(json!({
                        "id": "chatcmpl-123",
                        "object": "chat.completion",
                        "created": 1,
                        "model": "gpt-test",
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "hello from hub"
                            },
                            "finish_reason": "stop"
                        }],
                        "usage": {
                            "prompt_tokens": 7,
                            "completion_tokens": 9,
                            "total_tokens": 16,
                            "prompt_tokens_details": {"cached_tokens": 2},
                            "completion_tokens_details": {"audio_tokens": 1}
                        }
                    }))
                }
            }),
        );
        let (base_url, server) = spawn_server(router).await;

        let gateway = Gateway::new(ProviderRegistry::builder().build());
        assert!(gateway.registry().get("hub-test").is_none());

        let instance = ProviderInstance {
            def: Arc::new(HubTestProvider),
            auth: ProviderAuth::ApiKey("hub-secret".into()),
            base_url_override: Some(base_url),
            custom_headers: HeaderMap::new(),
        };
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "gpt-test",
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .unwrap();

        let response = gateway.chat_completion(&request, &instance).await.unwrap();
        let ChatResponse::Complete { response, usage } = response else {
            panic!("expected complete response")
        };

        assert_eq!(response.model, "gpt-test");
        assert!(matches!(
            response.choices[0].message.content.as_ref(),
            Some(crate::gateway::types::openai::MessageContent::Text(text))
                if text == "hello from hub"
        ));
        assert_eq!(usage.input_tokens, Some(7));
        assert_eq!(usage.output_tokens, Some(9));
        assert_eq!(usage.total_tokens, Some(16));
        assert_eq!(usage.cache_read_input_tokens, Some(2));
        assert_eq!(usage.output_audio_tokens, Some(1));

        let observed = observed.lock().await.take().unwrap();
        assert_eq!(observed.0.as_deref(), Some("Bearer hub-secret"));
        assert_eq!(observed.1["model"], "gpt-test");
        assert_eq!(observed.1["messages"][0]["content"], "hello");

        server.abort();
    }

    #[tokio::test]
    async fn chat_uses_native_path_when_format_and_provider_support_it() {
        let observed: Arc<Mutex<ObservedRequest>> = Arc::new(Mutex::new(None));
        let observed_clone = Arc::clone(&observed);
        let router = Router::new().route(
            "/v1/native",
            post(move |headers: HeaderMap, Json(body): Json<Value>| {
                let observed = Arc::clone(&observed_clone);
                async move {
                    let auth = headers
                        .get(AUTHORIZATION)
                        .and_then(|value| value.to_str().ok())
                        .map(str::to_owned);
                    *observed.lock().await = Some((auth, body));

                    Json(json!({
                        "ok": true,
                        "source": "native"
                    }))
                }
            }),
        );
        let (base_url, server) = spawn_server(router).await;

        let gateway = Gateway::new(ProviderRegistry::builder().build());
        let instance = ProviderInstance {
            def: Arc::new(NativeTestProvider),
            auth: ProviderAuth::ApiKey("native-secret".into()),
            base_url_override: Some(base_url),
            custom_headers: HeaderMap::new(),
        };
        let request = json!({
            "model": "native-model",
            "input": "hello"
        });

        let response = gateway
            .chat::<DummyNativeFormat>(&request, &instance)
            .await
            .unwrap();
        let ChatResponse::Complete { response, usage } = response else {
            panic!("expected complete response")
        };

        assert_eq!(response, json!({"ok": true, "source": "native"}));
        assert!(usage.input_tokens.is_none());
        assert!(usage.output_tokens.is_none());

        let observed = observed.lock().await.take().unwrap();
        assert_eq!(observed.0.as_deref(), Some("Bearer native-secret"));
        assert_eq!(observed.1, request);

        server.abort();
    }

    #[tokio::test]
    async fn chat_completion_streams_hub_chunks_and_reports_usage() {
        let sse_body = format!(
            "data: {}\n\ndata: {}\n\ndata: [DONE]\n\n",
            serde_json::to_string(&json!({
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "gpt-test",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": "hello from stream"
                    },
                    "finish_reason": null
                }]
            }))
            .unwrap(),
            serde_json::to_string(&json!({
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "gpt-test",
                "choices": [],
                "usage": {
                    "prompt_tokens": 7,
                    "completion_tokens": 9,
                    "total_tokens": 16
                }
            }))
            .unwrap(),
        );
        let router = Router::new().route(
            "/v1/chat/completions",
            post(move || {
                let sse_body = sse_body.clone();
                async move {
                    http::Response::builder()
                        .status(StatusCode::OK)
                        .header(CONTENT_TYPE, "text/event-stream")
                        .body(axum::body::Body::from(sse_body))
                        .unwrap()
                }
            }),
        );
        let (base_url, server) = spawn_server(router).await;

        let gateway = Gateway::new(ProviderRegistry::builder().build());
        let instance = ProviderInstance {
            def: Arc::new(HubTestProvider),
            auth: ProviderAuth::ApiKey("hub-secret".into()),
            base_url_override: Some(base_url),
            custom_headers: HeaderMap::new(),
        };
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "gpt-test",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": true
        }))
        .unwrap();

        let response = gateway.chat_completion(&request, &instance).await.unwrap();
        let ChatResponse::Stream {
            mut stream,
            usage_rx,
        } = response
        else {
            panic!("expected streaming response")
        };

        let first = stream.next().await.unwrap().unwrap();
        let usage_chunk = stream.next().await.unwrap().unwrap();
        assert!(stream.next().await.is_none());

        assert_eq!(
            first.choices[0].delta.content.as_deref(),
            Some("hello from stream")
        );
        assert_eq!(usage_chunk.usage.as_ref().unwrap().total_tokens, 16);

        let usage = usage_rx.await.unwrap();
        assert_eq!(usage.input_tokens, Some(7));
        assert_eq!(usage.output_tokens, Some(9));
        assert_eq!(usage.total_tokens, Some(16));

        server.abort();
    }

    #[tokio::test]
    async fn chat_streams_native_chunks_and_reports_usage() {
        let router = Router::new().route(
            "/v1/native-stream",
            post(|| async {
                http::Response::builder()
                    .status(StatusCode::OK)
                    .header(CONTENT_TYPE, "text/event-stream")
                    .body(axum::body::Body::from("data: buffered\n\ndata: usage\n\n"))
                    .unwrap()
            }),
        );
        let (base_url, server) = spawn_server(router).await;

        let gateway = Gateway::new(ProviderRegistry::builder().build());
        let instance = ProviderInstance {
            def: Arc::new(NativeTestProvider),
            auth: ProviderAuth::ApiKey("native-secret".into()),
            base_url_override: Some(base_url),
            custom_headers: HeaderMap::new(),
        };
        let request = json!({
            "model": "native-model",
            "stream": true
        });

        let response = gateway
            .chat::<StreamingNativeFormat>(&request, &instance)
            .await
            .unwrap();
        let ChatResponse::Stream {
            mut stream,
            usage_rx,
        } = response
        else {
            panic!("expected streaming response")
        };

        assert_eq!(
            stream.next().await.unwrap().unwrap(),
            json!({"value": "first"})
        );
        assert_eq!(
            stream.next().await.unwrap().unwrap(),
            json!({"value": "second"})
        );
        assert!(stream.next().await.is_none());

        let usage = usage_rx.await.unwrap();
        assert_eq!(usage.input_tokens, Some(5));
        assert_eq!(usage.output_tokens, Some(8));
        assert_eq!(usage.total_tokens, Some(13));

        server.abort();
    }

    #[tokio::test]
    async fn chat_completion_rejects_unsupported_stream_reader_before_dispatch() {
        let request_count = Arc::new(AtomicUsize::new(0));
        let request_count_clone = Arc::clone(&request_count);
        let router = Router::new().route(
            "/v1/chat/completions",
            post(move || {
                let request_count = Arc::clone(&request_count_clone);
                async move {
                    request_count.fetch_add(1, Ordering::SeqCst);
                    http::Response::builder()
                        .status(StatusCode::OK)
                        .header(CONTENT_TYPE, "text/event-stream")
                        .body(axum::body::Body::from("data: [DONE]\n\n"))
                        .unwrap()
                }
            }),
        );
        let (base_url, server) = spawn_server(router).await;

        let gateway = Gateway::new(ProviderRegistry::builder().build());
        let instance = ProviderInstance {
            def: Arc::new(UnsupportedHubStreamTestProvider),
            auth: ProviderAuth::ApiKey("hub-secret".into()),
            base_url_override: Some(base_url),
            custom_headers: HeaderMap::new(),
        };
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "gpt-test",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": true
        }))
        .unwrap();

        let result = gateway.chat_completion(&request, &instance).await;
        assert!(matches!(
            result,
            Err(GatewayError::Validation(message))
                if message.contains("JsonArrayStream")
        ));
        assert_eq!(request_count.load(Ordering::SeqCst), 0);

        server.abort();
    }

    #[tokio::test]
    async fn chat_native_rejects_unsupported_stream_reader_before_dispatch() {
        let request_count = Arc::new(AtomicUsize::new(0));
        let request_count_clone = Arc::clone(&request_count);
        let router = Router::new().route(
            "/v1/native-stream",
            post(move || {
                let request_count = Arc::clone(&request_count_clone);
                async move {
                    request_count.fetch_add(1, Ordering::SeqCst);
                    http::Response::builder()
                        .status(StatusCode::OK)
                        .header(CONTENT_TYPE, "text/event-stream")
                        .body(axum::body::Body::from("data: [DONE]\n\n"))
                        .unwrap()
                }
            }),
        );
        let (base_url, server) = spawn_server(router).await;

        let gateway = Gateway::new(ProviderRegistry::builder().build());
        let instance = ProviderInstance {
            def: Arc::new(UnsupportedNativeStreamTestProvider),
            auth: ProviderAuth::ApiKey("native-secret".into()),
            base_url_override: Some(base_url),
            custom_headers: HeaderMap::new(),
        };
        let request = json!({
            "model": "native-model",
            "stream": true
        });

        let result = gateway
            .chat::<StreamingNativeFormat>(&request, &instance)
            .await;
        assert!(matches!(
            result,
            Err(GatewayError::Validation(message))
                if message.contains("AwsEventStream")
        ));
        assert_eq!(request_count.load(Ordering::SeqCst), 0);

        server.abort();
    }

    #[tokio::test]
    async fn chat_completion_preserves_non_json_provider_error_body() {
        let router = Router::new().route(
            "/v1/chat/completions",
            post(|| async { (StatusCode::BAD_GATEWAY, "upstream exploded") }),
        );
        let (base_url, server) = spawn_server(router).await;

        let gateway = Gateway::new(ProviderRegistry::builder().build());
        let instance = ProviderInstance {
            def: Arc::new(HubTestProvider),
            auth: ProviderAuth::ApiKey("hub-secret".into()),
            base_url_override: Some(base_url),
            custom_headers: HeaderMap::new(),
        };
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "gpt-test",
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .unwrap();

        let result = gateway.chat_completion(&request, &instance).await;
        match result {
            Err(GatewayError::Provider {
                status,
                body,
                provider,
                retryable,
            }) => {
                assert_eq!(status, StatusCode::BAD_GATEWAY);
                assert_eq!(body, Value::String("upstream exploded".into()));
                assert_eq!(provider, "hub-test");
                assert!(retryable);
            }
            Err(other) => panic!("unexpected gateway error: {other}"),
            Ok(_) => panic!("expected provider error"),
        }

        server.abort();
    }

    fn bearer_headers(provider: &str, auth: &ProviderAuth) -> Result<HeaderMap> {
        let mut headers = HeaderMap::new();
        let value = HeaderValue::from_str(&format!("Bearer {}", auth.api_key_for(provider)?))
            .map_err(|error| GatewayError::Validation(error.to_string()))?;
        headers.insert(AUTHORIZATION, value);
        headers.insert(
            HeaderName::from_static("x-provider-name"),
            HeaderValue::from_str(provider)
                .map_err(|error| GatewayError::Validation(error.to_string()))?,
        );
        Ok(headers)
    }

    async fn spawn_server(router: Router) -> (Url, JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });
        let base_url = Url::parse(&format!("http://{addr}")).unwrap();
        (base_url, server)
    }
}
