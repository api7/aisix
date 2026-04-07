use http::StatusCode;
use serde_json::Value;

use crate::gateway::{
    error::{GatewayError, Result},
    formats::OpenAIChatFormat,
    provider_instance::{ProviderInstance, ProviderRegistry},
    traits::{ChatFormat, NativeHandler},
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
    /// Creates a new gateway with the provided provider registry.
    pub fn new(registry: ProviderRegistry) -> Self {
        Self {
            registry,
            http_client: reqwest::Client::new(),
        }
    }

    /// Returns the immutable provider registry backing this gateway.
    pub fn registry(&self) -> &ProviderRegistry {
        &self.registry
    }

    /// Non-streaming typed chat entry point.
    pub async fn chat<F: ChatFormat>(
        &self,
        request: &F::Request,
        instance: &ProviderInstance,
    ) -> Result<ChatResponse<F>> {
        if F::is_stream(request) {
            return Err(GatewayError::Validation(format!(
                "streaming requests for format {} are not implemented yet",
                F::name()
            )));
        }

        if let Some(native) = F::native_support(instance.def.as_ref()) {
            return self.call_chat_native::<F>(&native, instance, request).await;
        }

        let (hub_request, ctx) = F::to_hub(request)?;
        let hub_response = self.call_chat_hub(instance, &hub_request).await?;
        let usage = extract_chat_usage_from_response(&hub_response).unwrap_or_default();
        let response = F::from_hub(&hub_response, &ctx)?;

        Ok(ChatResponse::Complete { response, usage })
    }

    /// Convenience wrapper for the OpenAI Chat format.
    pub async fn chat_completion(
        &self,
        request: &ChatCompletionRequest,
        instance: &ProviderInstance,
    ) -> Result<ChatResponse<OpenAIChatFormat>> {
        self.chat::<OpenAIChatFormat>(request, instance).await
    }

    async fn call_chat_native<F: ChatFormat>(
        &self,
        native: &NativeHandler<'_>,
        instance: &ProviderInstance,
        request: &F::Request,
    ) -> Result<ChatResponse<F>> {
        let (endpoint_path, body) = F::call_native(native, request, false)?;
        let base_url = instance.effective_base_url()?;
        let url = join_url(base_url.as_str(), &endpoint_path);
        let headers = instance.build_headers()?;

        let response = self
            .http_client
            .post(url)
            .headers(headers)
            .json(&body)
            .send()
            .await
            .map_err(GatewayError::Http)?;

        if !response.status().is_success() {
            return Err(provider_error(response, instance.def.name()).await);
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

        let response = self
            .http_client
            .post(url)
            .headers(headers)
            .json(&provider_body)
            .send()
            .await
            .map_err(GatewayError::Http)?;

        if !response.status().is_success() {
            return Err(provider_error(response, instance.def.name()).await);
        }

        let body: Value = response.json().await.map_err(GatewayError::Http)?;
        instance.def.transform_response(body)
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
    let body = response.json().await.unwrap_or(Value::Null);

    GatewayError::Provider {
        status,
        body,
        provider: provider.to_string(),
        retryable: status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error(),
    }
}

#[cfg(test)]
mod tests {
    use std::{borrow::Cow, sync::Arc};

    use axum::{Json, Router, routing::post};
    use http::{
        HeaderMap, HeaderValue,
        header::{AUTHORIZATION, HeaderName},
    };
    use reqwest::Url;
    use serde_json::{Value, json};
    use tokio::{net::TcpListener, sync::Mutex, task::JoinHandle};

    use super::Gateway;
    use crate::gateway::{
        error::{GatewayError, Result},
        formats::OpenAIChatFormat,
        provider_instance::{ProviderAuth, ProviderInstance, ProviderRegistry},
        traits::{
            ChatFormat, ChatTransform, NativeHandler, NativeOpenAIResponsesSupport,
            OpenAIResponsesNativeStreamState, ProviderCapabilities, ProviderMeta, StreamReaderKind,
        },
        types::{
            common::BridgeContext,
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

    struct DummyNativeFormat;

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
    async fn chat_rejects_streaming_requests_until_pr_4_2() {
        let gateway = Gateway::new(ProviderRegistry::builder().build());
        let instance = ProviderInstance {
            def: Arc::new(HubTestProvider),
            auth: ProviderAuth::None,
            base_url_override: None,
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
                if message.contains("streaming requests") && message.contains(OpenAIChatFormat::name())
        ));
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
