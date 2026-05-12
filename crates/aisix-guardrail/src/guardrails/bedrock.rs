use std::{fmt, time::SystemTime};

use async_trait::async_trait;
use aws_credential_types::Credentials;
use aws_sigv4::{
    http_request::{SignableBody, SignableRequest, SigningSettings, sign},
    sign::v4,
};
use aws_smithy_runtime_api::client::identity::Identity;
use http::{HeaderMap, HeaderValue, Method, Request, StatusCode, header::CONTENT_TYPE};
use percent_encoding::{AsciiSet, CONTROLS, utf8_percent_encode};
use reqwest::{Client, Url};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use utoipa::ToSchema;

use crate::traits::{
    GuardrailCheckPayload, GuardrailMessage, GuardrailMessageContent, GuardrailMeta,
    GuardrailOutcome, GuardrailRuntime, GuardrailStage,
};

pub const IDENTIFIER: &str = "bedrock";

const DEFAULT_OUTPUT_SCOPE: &str = "INTERVENTIONS";
const DEFAULT_RUNTIME_HOST_PREFIX: &str = "https://bedrock-runtime.";
const DEFAULT_RUNTIME_HOST_SUFFIX: &str = ".amazonaws.com";
type EncodedApplyGuardrailRequest = (Vec<u8>, Vec<usize>);

const PATH_SEGMENT_ENCODE_SET: &AsciiSet = &CONTROLS
    .add(b' ')
    .add(b'"')
    .add(b'#')
    .add(b'%')
    .add(b'<')
    .add(b'>')
    .add(b'?')
    .add(b'`')
    .add(b'{')
    .add(b'}')
    .add(b'/');

#[derive(Clone, Serialize, Deserialize, ToSchema)]
pub struct BedrockGuardrailConfig {
    pub identifier: String,
    pub version: String,
    pub region: String,
    pub access_key_id: String,
    pub secret_access_key: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_token: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<String>,
}

impl fmt::Debug for BedrockGuardrailConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const REDACTED: &str = "[REDACTED]";

        f.debug_struct("BedrockGuardrailConfig")
            .field("identifier", &self.identifier)
            .field("version", &self.version)
            .field("region", &self.region)
            .field("access_key_id", &REDACTED)
            .field("secret_access_key", &REDACTED)
            .field(
                "session_token",
                &self.session_token.as_ref().map(|_| REDACTED),
            )
            .field("endpoint", &self.endpoint)
            .finish()
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct BedrockGuardrailMeta;

impl GuardrailMeta for BedrockGuardrailMeta {
    fn name(&self) -> &'static str {
        IDENTIFIER
    }
}

#[derive(Debug, Clone)]
pub struct BedrockGuardrailRuntime {
    client: Client,
}

impl Default for BedrockGuardrailRuntime {
    fn default() -> Self {
        Self::new()
    }
}

impl BedrockGuardrailRuntime {
    pub fn new() -> Self {
        Self {
            client: Client::new(),
        }
    }

    pub fn with_client(client: Client) -> Self {
        Self { client }
    }

    fn build_apply_url(&self, config: &BedrockGuardrailConfig) -> Result<Url, BedrockError> {
        let base_url = config
            .endpoint
            .as_deref()
            .map(str::to_owned)
            .unwrap_or_else(|| {
                format!(
                    "{DEFAULT_RUNTIME_HOST_PREFIX}{}{DEFAULT_RUNTIME_HOST_SUFFIX}",
                    config.region
                )
            });
        let endpoint_path = format!(
            "/guardrail/{}/version/{}/apply",
            encode_path_segment(&config.identifier),
            encode_path_segment(&config.version),
        );

        build_url_for_endpoint(&base_url, &endpoint_path)
    }

    fn build_request_body(
        &self,
        payload: &GuardrailCheckPayload,
    ) -> Result<Option<EncodedApplyGuardrailRequest>, BedrockError> {
        let (content, message_indexes) = text_blocks_from_payload(payload);
        if content.is_empty() {
            return Ok(None);
        }

        let body = ApplyGuardrailRequest {
            content,
            output_scope: Some(DEFAULT_OUTPUT_SCOPE),
            source: match payload.stage() {
                GuardrailStage::Input => "INPUT",
                GuardrailStage::Output => "OUTPUT",
            },
        };

        Ok(Some((serde_json::to_vec(&body)?, message_indexes)))
    }

    fn sign_request(
        &self,
        url: &Url,
        body: &[u8],
        time: SystemTime,
        config: &BedrockGuardrailConfig,
    ) -> Result<SignedRequest, BedrockError> {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        let header_pairs = headers
            .iter()
            .map(|(name, value)| {
                let value = value.to_str().map_err(|error| {
                    BedrockError::Signing(format!(
                        "bedrock guardrail produced non-utf8 header {}: {}",
                        name, error
                    ))
                })?;
                Ok((name.as_str().to_owned(), value.to_owned()))
            })
            .collect::<Result<Vec<_>, BedrockError>>()?;

        let identity: Identity = Credentials::new(
            config.access_key_id.clone(),
            config.secret_access_key.clone(),
            config.session_token.clone(),
            None,
            "aisix-bedrock-guardrail-static",
        )
        .into();
        let signing_params = v4::SigningParams::builder()
            .identity(&identity)
            .region(config.region.as_str())
            .name("bedrock")
            .time(time)
            .settings(SigningSettings::default())
            .build()
            .map_err(|error| BedrockError::Signing(error.to_string()))?
            .into();
        let signable_request = SignableRequest::new(
            Method::POST.as_str(),
            url.as_str(),
            header_pairs
                .iter()
                .map(|(name, value)| (name.as_str(), value.as_str())),
            SignableBody::Bytes(body),
        )
        .map_err(|error| BedrockError::Signing(error.to_string()))?;

        let mut signed_request = Request::builder().method(Method::POST).uri(url.as_str());
        for (name, value) in &headers {
            signed_request = signed_request.header(name, value);
        }

        let mut signed_request = signed_request
            .body(())
            .map_err(|error| BedrockError::Signing(error.to_string()))?;
        let (instructions, _signature) = sign(signable_request, &signing_params)
            .map_err(|error| BedrockError::Signing(error.to_string()))?
            .into_parts();
        instructions.apply_to_request_http1x(&mut signed_request);

        Ok(SignedRequest {
            url: Url::parse(&signed_request.uri().to_string())
                .map_err(|error| BedrockError::Signing(error.to_string()))?,
            headers: signed_request.headers().clone(),
        })
    }
}

impl GuardrailMeta for BedrockGuardrailRuntime {
    fn name(&self) -> &'static str {
        IDENTIFIER
    }
}

#[async_trait]
impl GuardrailRuntime<BedrockGuardrailConfig> for BedrockGuardrailRuntime {
    type Error = BedrockError;

    async fn check(
        &self,
        payload: &GuardrailCheckPayload,
        config: &BedrockGuardrailConfig,
    ) -> Result<GuardrailOutcome, Self::Error> {
        let Some((body, message_indexes)) = self.build_request_body(payload)? else {
            return Ok(GuardrailOutcome::Allow);
        };

        let url = self.build_apply_url(config)?;
        let signed = self.sign_request(&url, &body, SystemTime::now(), config)?;
        let response = self
            .client
            .post(signed.url)
            .headers(signed.headers)
            .body(body)
            .send()
            .await?;
        let status = response.status();
        let response_body = response.bytes().await?;

        if !status.is_success() {
            let body = String::from_utf8_lossy(&response_body).into_owned();
            return Err(BedrockError::HttpStatus(status, body));
        }

        let response: ApplyGuardrailResponse = serde_json::from_slice(&response_body)?;
        outcome_from_response(payload, &message_indexes, response)
    }
}

#[derive(Debug, Error)]
pub enum BedrockError {
    #[error("bedrock guardrail payload is not supported: {0}")]
    UnsupportedPayload(String),
    #[error("failed to build bedrock guardrail url: {0}")]
    Url(String),
    #[error("failed to sign bedrock guardrail request: {0}")]
    Signing(String),
    #[error("bedrock guardrail request failed: {0}")]
    Request(#[from] reqwest::Error),
    #[error("failed to encode/decode bedrock guardrail json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("bedrock guardrail returned HTTP {0}: {1}")]
    HttpStatus(StatusCode, String),
}

#[derive(Debug, Clone)]
struct SignedRequest {
    url: Url,
    headers: HeaderMap,
}

#[derive(Debug, Serialize)]
struct ApplyGuardrailRequest {
    content: Vec<ApplyGuardrailContentBlock>,
    #[serde(rename = "outputScope", skip_serializing_if = "Option::is_none")]
    output_scope: Option<&'static str>,
    source: &'static str,
}

#[derive(Debug, Serialize)]
struct ApplyGuardrailContentBlock {
    text: ApplyGuardrailTextBlock,
}

#[derive(Debug, Serialize)]
struct ApplyGuardrailTextBlock {
    text: String,
}

#[derive(Debug, Deserialize)]
struct ApplyGuardrailResponse {
    action: String,
    #[serde(rename = "actionReason")]
    action_reason: Option<String>,
    #[serde(default)]
    outputs: Vec<ApplyGuardrailOutput>,
}

#[derive(Debug, Deserialize)]
struct ApplyGuardrailOutput {
    text: String,
}

fn encode_path_segment(segment: &str) -> String {
    utf8_percent_encode(segment, PATH_SEGMENT_ENCODE_SET).to_string()
}

fn build_url_for_endpoint(base_url: &str, endpoint_path: &str) -> Result<Url, BedrockError> {
    let mut parsed = Url::parse(base_url).map_err(|error| BedrockError::Url(error.to_string()))?;

    let base_segments = parsed
        .path_segments()
        .map(|segments| {
            segments
                .filter(|segment| !segment.is_empty())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let endpoint_segments = endpoint_path
        .split('/')
        .filter(|segment| !segment.is_empty())
        .collect::<Vec<_>>();

    let max_overlap = base_segments.len().min(endpoint_segments.len());
    let overlap = (1..=max_overlap)
        .rev()
        .find(|count| base_segments[base_segments.len() - count..] == endpoint_segments[..*count])
        .unwrap_or(0);

    let mut joined_segments = base_segments;
    joined_segments.extend_from_slice(&endpoint_segments[overlap..]);

    parsed.set_path(&format!("/{}", joined_segments.join("/")));
    Ok(parsed)
}

fn text_blocks_from_payload(
    payload: &GuardrailCheckPayload,
) -> (Vec<ApplyGuardrailContentBlock>, Vec<usize>) {
    let messages = match payload {
        GuardrailCheckPayload::Input(payload) => &payload.messages,
        GuardrailCheckPayload::Output(payload) => &payload.messages,
    };

    messages
        .iter()
        .enumerate()
        .filter_map(|(index, message)| message_text_for_bedrock(message).map(|text| (index, text)))
        .fold(
            (Vec::new(), Vec::new()),
            |(mut content, mut indexes), (index, text)| {
                indexes.push(index);
                content.push(ApplyGuardrailContentBlock {
                    text: ApplyGuardrailTextBlock { text },
                });
                (content, indexes)
            },
        )
}

fn message_text_for_bedrock(message: &GuardrailMessage) -> Option<String> {
    match &message.content {
        Some(GuardrailMessageContent::Text(text)) if !text.is_empty() => Some(text.clone()),
        _ => None,
    }
}

fn outcome_from_response(
    payload: &GuardrailCheckPayload,
    message_indexes: &[usize],
    response: ApplyGuardrailResponse,
) -> Result<GuardrailOutcome, BedrockError> {
    match response.action.as_str() {
        "NONE" => Ok(GuardrailOutcome::Allow),
        "GUARDRAIL_BLOCKED" => Ok(GuardrailOutcome::Block {
            reason: response
                .action_reason
                .unwrap_or_else(|| "bedrock guardrail blocked".into()),
        }),
        "GUARDRAIL_INTERVENED" => {
            if response.outputs.is_empty() {
                return Ok(GuardrailOutcome::Block {
                    reason: response
                        .action_reason
                        .unwrap_or_else(|| "bedrock guardrail intervened".into()),
                });
            }

            if response.outputs.len() != message_indexes.len() {
                return Ok(GuardrailOutcome::Block {
                    reason: format!(
                        "bedrock guardrail returned {} rewritten outputs for {} text messages",
                        response.outputs.len(),
                        message_indexes.len()
                    ),
                });
            }

            let mut rewritten = payload.clone();
            let messages = match &mut rewritten {
                GuardrailCheckPayload::Input(payload) => &mut payload.messages,
                GuardrailCheckPayload::Output(payload) => &mut payload.messages,
            };

            for (index, output) in message_indexes.iter().zip(response.outputs) {
                messages[*index].content = Some(GuardrailMessageContent::Text(output.text));
            }

            Ok(GuardrailOutcome::Rewrite(rewritten))
        }
        other => Err(BedrockError::UnsupportedPayload(format!(
            "unknown bedrock guardrail action {other}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, UNIX_EPOCH};

    use reqwest::Url;

    use super::{
        ApplyGuardrailOutput, ApplyGuardrailResponse, BedrockError, BedrockGuardrailConfig,
        BedrockGuardrailRuntime, build_url_for_endpoint, outcome_from_response,
    };
    use crate::traits::{
        GuardrailCheckPayload, GuardrailMessage, GuardrailMessageContent, GuardrailOutcome,
        GuardrailRole, InputGuardrailPayload,
    };

    fn config() -> BedrockGuardrailConfig {
        BedrockGuardrailConfig {
            identifier: "guardrail-123".into(),
            version: "1".into(),
            region: "us-east-1".into(),
            access_key_id: "AKIA123".into(),
            secret_access_key: "secret".into(),
            session_token: Some("token".into()),
            endpoint: Some("https://bedrock-runtime.us-east-1.amazonaws.com/guardrail".into()),
        }
    }

    fn runtime() -> BedrockGuardrailRuntime {
        BedrockGuardrailRuntime::new()
    }

    fn input_payload(text: &str) -> GuardrailCheckPayload {
        GuardrailCheckPayload::Input(InputGuardrailPayload {
            messages: vec![GuardrailMessage {
                role: GuardrailRole::User,
                content: Some(GuardrailMessageContent::Text(text.into())),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            }],
        })
    }

    #[test]
    fn bedrock_guardrail_config_debug_redacts_credentials() {
        let config = config();

        let output = format!("{config:?}");
        assert!(output.contains("[REDACTED]"));
        assert!(!output.contains("AKIA123"));
        assert!(!output.contains("secret_access_key: \"secret\""));
        assert!(!output.contains("session_token: Some(\"token\")"));
    }

    #[test]
    fn build_url_for_endpoint_handles_overlap_and_encoding() {
        let url = build_url_for_endpoint(
            "https://bedrock-runtime.us-east-1.amazonaws.com/guardrail",
            "/guardrail/arn:aws:bedrock:us-east-1:123456789012:guardrail/my/guardrail/version/DRAFT/apply",
        )
        .unwrap();

        assert_eq!(
            url.as_str(),
            "https://bedrock-runtime.us-east-1.amazonaws.com/guardrail/arn:aws:bedrock:us-east-1:123456789012:guardrail/my/guardrail/version/DRAFT/apply"
        );
    }

    #[test]
    fn build_apply_url_percent_encodes_identifier_segments() {
        let runtime = runtime();
        let mut config = config();
        config.identifier = "guardrail/name".into();
        config.version = "DRAFT".into();
        let url = runtime.build_apply_url(&config).unwrap();

        assert!(
            url.path()
                .ends_with("/guardrail/guardrail%2Fname/version/DRAFT/apply")
        );
    }

    #[test]
    fn sign_request_adds_sigv4_authorization_header() {
        let runtime = runtime();
        let config = config();
        let signed = runtime
            .sign_request(
                &Url::parse(
                    "https://bedrock-runtime.us-east-1.amazonaws.com/guardrail/gr/version/1/apply",
                )
                .unwrap(),
                br#"{"content":[{"text":{"text":"hello"}}],"source":"INPUT"}"#,
                UNIX_EPOCH + Duration::from_secs(1_700_000_000),
                &config,
            )
            .unwrap();

        assert!(
            signed
                .headers
                .get(http::header::AUTHORIZATION)
                .and_then(|value| value.to_str().ok())
                .is_some_and(|value| value.starts_with("AWS4-HMAC-SHA256"))
        );
    }

    #[test]
    fn outcome_from_response_allows_when_no_intervention() {
        let outcome = outcome_from_response(
            &input_payload("hello"),
            &[0],
            ApplyGuardrailResponse {
                action: "NONE".into(),
                action_reason: None,
                outputs: vec![],
            },
        )
        .unwrap();

        assert_eq!(outcome, GuardrailOutcome::Allow);
    }

    #[test]
    fn outcome_from_response_rewrites_text_messages() {
        let outcome = outcome_from_response(
            &input_payload("hello"),
            &[0],
            ApplyGuardrailResponse {
                action: "GUARDRAIL_INTERVENED".into(),
                action_reason: Some("filtered".into()),
                outputs: vec![ApplyGuardrailOutput {
                    text: "hi there".into(),
                }],
            },
        )
        .unwrap();

        assert_eq!(
            outcome,
            GuardrailOutcome::Rewrite(GuardrailCheckPayload::Input(InputGuardrailPayload {
                messages: vec![GuardrailMessage {
                    role: GuardrailRole::User,
                    content: Some(GuardrailMessageContent::Text("hi there".into())),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                }],
            }))
        );
    }

    #[test]
    fn outcome_from_response_blocks_when_intervention_has_no_rewrite_output() {
        let outcome = outcome_from_response(
            &input_payload("hello"),
            &[0],
            ApplyGuardrailResponse {
                action: "GUARDRAIL_INTERVENED".into(),
                action_reason: Some("policy triggered".into()),
                outputs: vec![],
            },
        )
        .unwrap();

        assert_eq!(
            outcome,
            GuardrailOutcome::Block {
                reason: "policy triggered".into(),
            }
        );
    }

    #[test]
    fn outcome_from_response_blocks_when_bedrock_reports_blocked() {
        let outcome = outcome_from_response(
            &input_payload("hello"),
            &[0],
            ApplyGuardrailResponse {
                action: "GUARDRAIL_BLOCKED".into(),
                action_reason: Some("policy triggered".into()),
                outputs: vec![],
            },
        )
        .unwrap();

        assert_eq!(
            outcome,
            GuardrailOutcome::Block {
                reason: "policy triggered".into(),
            }
        );
    }

    #[test]
    fn outcome_from_response_rejects_unknown_actions() {
        let error = outcome_from_response(
            &input_payload("hello"),
            &[0],
            ApplyGuardrailResponse {
                action: "MAYBE".into(),
                action_reason: None,
                outputs: vec![],
            },
        )
        .unwrap_err();

        assert!(matches!(error, BedrockError::UnsupportedPayload(_)));
    }
}
