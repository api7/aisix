mod transform;

use std::{borrow::Cow, time::SystemTime};

use aws_credential_types::Credentials;
use aws_sigv4::{
    http_request::{SignableBody, SignableRequest, SigningSettings, sign},
    sign::v4,
};
use aws_smithy_runtime_api::client::identity::Identity;
use http::{HeaderMap, Request};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::gateway::{
    error::{GatewayError, Result},
    provider_instance::ProviderAuth,
    traits::{
        ChatStreamState, ChatTransform, PreparedRequest, ProviderCapabilities, ProviderMeta,
        StreamReaderKind,
    },
    types::openai::{ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse},
};

pub const IDENTIFIER: &str = "bedrock";

const DEFAULT_BASE_URL: &str = "https://bedrock-runtime.us-east-1.amazonaws.com";

pub struct BedrockDef;

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct BedrockProviderConfig {
    pub region: String,
    pub access_key_id: String,
    pub secret_access_key: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_token: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<String>,
}

impl ProviderMeta for BedrockDef {
    fn name(&self) -> &'static str {
        IDENTIFIER
    }

    fn default_base_url(&self) -> &'static str {
        DEFAULT_BASE_URL
    }

    fn chat_endpoint_path(&self, _model: &str) -> Cow<'static, str> {
        Cow::Borrowed("/model")
    }

    fn stream_reader_kind(&self) -> StreamReaderKind {
        StreamReaderKind::AwsEventStream
    }

    fn prepare_request(
        &self,
        mut request: PreparedRequest,
        auth: &ProviderAuth,
    ) -> Result<PreparedRequest> {
        if request.stream {
            let mut path = request.url.path().to_string();
            let Some(prefix) = path.strip_suffix("/converse") else {
                return Err(GatewayError::Validation(format!(
                    "provider {} expected a converse path before stream signing, got {}",
                    self.name(),
                    path
                )));
            };
            path = format!("{prefix}/converse-stream");
            request.url.set_path(&path);
        }

        let aws = auth.aws_static_credentials_for(self.name())?;
        let header_pairs = request
            .headers
            .iter()
            .map(|(name, value)| {
                let value = value.to_str().map_err(|error| {
                    GatewayError::Validation(format!(
                        "provider {} produced non-utf8 header {}: {}",
                        self.name(),
                        name,
                        error
                    ))
                })?;
                Ok((name.as_str().to_owned(), value.to_owned()))
            })
            .collect::<Result<Vec<_>>>()?;

        let identity: Identity = Credentials::new(
            aws.access_key_id.clone(),
            aws.secret_access_key.clone(),
            aws.session_token.clone(),
            None,
            "aisix-bedrock-static",
        )
        .into();
        let signing_params = v4::SigningParams::builder()
            .identity(&identity)
            .region(aws.region.as_str())
            .name("bedrock")
            .time(SystemTime::now())
            .settings(SigningSettings::default())
            .build()
            .map_err(|error| GatewayError::Internal(error.to_string()))?
            .into();
        let signable_request = SignableRequest::new(
            request.method.as_str(),
            request.url.as_str(),
            header_pairs
                .iter()
                .map(|(name, value)| (name.as_str(), value.as_str())),
            SignableBody::Bytes(&request.body),
        )
        .map_err(|error| GatewayError::Internal(error.to_string()))?;
        let mut signed_request = Request::builder()
            .method(request.method.clone())
            .uri(request.url.as_str());

        for (name, value) in &request.headers {
            signed_request = signed_request.header(name, value);
        }

        let mut signed_request = signed_request
            .body(())
            .map_err(|error| GatewayError::Internal(error.to_string()))?;
        let (instructions, _signature) = sign(signable_request, &signing_params)
            .map_err(|error| GatewayError::Internal(error.to_string()))?
            .into_parts();
        instructions.apply_to_request_http1x(&mut signed_request);

        request.url = reqwest::Url::parse(&signed_request.uri().to_string())
            .map_err(|error| GatewayError::Internal(error.to_string()))?;
        request.headers = signed_request.headers().clone();
        Ok(request)
    }

    fn build_auth_headers(&self, _auth: &ProviderAuth) -> Result<HeaderMap> {
        Ok(HeaderMap::new())
    }

    fn build_url(&self, base_url: &str, model: &str) -> String {
        let Ok(mut url) = reqwest::Url::parse(base_url) else {
            return format!(
                "{}/model/{}/converse",
                base_url.trim_end_matches('/'),
                model.replace('/', "%2F")
            );
        };

        if let Ok(mut segments) = url.path_segments_mut() {
            segments.pop_if_empty();
            segments.push("model");
            segments.push(model);
            segments.push("converse");
        }

        url.to_string()
    }
}

impl ChatTransform for BedrockDef {
    fn transform_request(&self, request: &ChatCompletionRequest) -> Result<Value> {
        serde_json::to_value(transform::openai_to_bedrock_request(request)?)
            .map_err(|error| GatewayError::Transform(error.to_string()))
    }

    fn transform_response(&self, _body: Value) -> Result<ChatCompletionResponse> {
        Err(GatewayError::Transform(
            "bedrock transform_response requires request context".into(),
        ))
    }

    fn transform_response_with_request(
        &self,
        request: &ChatCompletionRequest,
        body: Value,
    ) -> Result<ChatCompletionResponse> {
        let response = serde_json::from_value(body)
            .map_err(|error| GatewayError::Transform(error.to_string()))?;
        transform::bedrock_to_openai_response(request, response)
    }

    fn transform_stream_chunk(
        &self,
        raw: &str,
        state: &mut ChatStreamState,
    ) -> Result<Vec<ChatCompletionChunk>> {
        transform::parse_bedrock_stream_to_openai(raw, state)
    }
}

impl ProviderCapabilities for BedrockDef {}

#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use http::{HeaderMap, HeaderValue, Method, header::CONTENT_TYPE};
    use serde_json::json;

    use super::{BedrockDef, BedrockProviderConfig};
    use crate::gateway::{
        provider_instance::ProviderAuth,
        traits::{PreparedRequest, ProviderMeta},
    };

    #[test]
    fn bedrock_provider_config_deserializes_static_credentials() {
        let config: BedrockProviderConfig = serde_json::from_value(json!({
            "region": "us-east-1",
            "access_key_id": "AKIA123",
            "secret_access_key": "secret",
            "session_token": "token",
            "endpoint": "https://bedrock-runtime.us-east-1.amazonaws.com"
        }))
        .unwrap();

        assert_eq!(config.region, "us-east-1");
        assert_eq!(config.access_key_id, "AKIA123");
        assert_eq!(config.secret_access_key, "secret");
        assert_eq!(config.session_token.as_deref(), Some("token"));
        assert_eq!(
            config.endpoint.as_deref(),
            Some("https://bedrock-runtime.us-east-1.amazonaws.com")
        );
    }

    #[test]
    fn build_url_encodes_model_ids_with_slashes() {
        let provider = BedrockDef;

        let url = provider.build_url(
            "https://bedrock-runtime.us-east-1.amazonaws.com",
            "inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        );

        assert_eq!(
            url,
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/inference-profile%2Fus.anthropic.claude-3-7-sonnet-20250219-v1:0/converse"
        );
    }

    #[test]
    fn prepare_request_rejects_non_aws_auth() {
        let provider = BedrockDef;
        let request = PreparedRequest {
            method: Method::POST,
            url: reqwest::Url::parse(
                "https://bedrock-runtime.us-east-1.amazonaws.com/model/test/converse",
            )
            .unwrap(),
            headers: {
                let mut headers = HeaderMap::new();
                headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
                headers
            },
            body: Bytes::from_static(b"{}"),
            stream: false,
        };

        let error = provider
            .prepare_request(request, &ProviderAuth::ApiKey("secret".into()))
            .unwrap_err();

        assert!(matches!(
            error,
            crate::gateway::error::GatewayError::Validation(message)
                if message.contains("ProviderAuth::AwsStatic")
        ));
    }

    #[test]
    fn prepare_request_rewrites_stream_url_before_signing() {
        let provider = BedrockDef;
        let request = PreparedRequest {
            method: Method::POST,
            url: reqwest::Url::parse(
                "https://bedrock-runtime.us-east-1.amazonaws.com/model/test/converse",
            )
            .unwrap(),
            headers: {
                let mut headers = HeaderMap::new();
                headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
                headers
            },
            body: Bytes::from_static(b"{}"),
            stream: true,
        };

        let prepared = provider
            .prepare_request(
                request,
                &ProviderAuth::AwsStatic(crate::gateway::provider_instance::AwsStaticCredentials {
                    access_key_id: "AKIA123".into(),
                    secret_access_key: "secret".into(),
                    session_token: None,
                    region: "us-east-1".into(),
                }),
            )
            .unwrap();

        assert_eq!(prepared.url.path(), "/model/test/converse-stream");
        assert!(
            prepared
                .headers
                .get(http::header::AUTHORIZATION)
                .and_then(|value| value.to_str().ok())
                .is_some_and(|value| value.starts_with("AWS4-HMAC-SHA256"))
        );
    }
}
