//! The Azure OpenAI URL, route and auth header for every operation the
//! gateway dispatches to an `azure-openai` Provider Key — chat completions
//! and embeddings from the bridge, images and audio from the proxy
//! handlers. One builder for the whole family, so the routes cannot drift
//! apart.
//!
//! Two routes, chosen by the key's `api_version`
//! (<https://learn.microsoft.com/en-us/azure/ai-foundry/openai/api-version-lifecycle>):
//!
//! - a dated version, or none: the deployment route
//!   `{base}/openai/deployments/{deployment}/{path}?api-version={version}`;
//! - `v1`, `preview` or `latest`: the v1 route `{base}/openai/v1/{path}`,
//!   with no `api-version` query, which routes by the body's `model`.

use aisix_core::ProviderKey;
use aisix_gateway::url_cache::{cached_model_endpoint_url, EndpointUrl};
use aisix_gateway::BridgeError;
use http::{HeaderName, HeaderValue};

use crate::bridge::{AzureCredential, AzureUpstreamRef};

/// An operation dispatched to Azure OpenAI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AzureOperation {
    ChatCompletions,
    Embeddings,
    AudioTranscriptions,
    AudioTranslations,
    AudioSpeech,
    ImageGenerations,
    ImageEdits,
}

impl AzureOperation {
    /// The path under the deployment (or v1) prefix.
    pub const fn path(self) -> &'static str {
        match self {
            Self::ChatCompletions => "chat/completions",
            Self::Embeddings => "embeddings",
            Self::AudioTranscriptions => "audio/transcriptions",
            Self::AudioTranslations => "audio/translations",
            Self::AudioSpeech => "audio/speech",
            Self::ImageGenerations => "images/generations",
            Self::ImageEdits => "images/edits",
        }
    }

    /// The dated version used when the key configures none. Speech and
    /// the image operations are not in the `2024-10-21` GA surface.
    pub const fn default_api_version(self) -> &'static str {
        match self {
            Self::ChatCompletions
            | Self::Embeddings
            | Self::AudioTranscriptions
            | Self::AudioTranslations => AzureUpstreamRef::DEFAULT_API_VERSION,
            Self::AudioSpeech | Self::ImageGenerations | Self::ImageEdits => "2025-04-01-preview",
        }
    }

    const fn url_cache_key(self) -> &'static str {
        match self {
            Self::ChatCompletions => "azure/chat",
            Self::Embeddings => "azure/embeddings",
            Self::AudioTranscriptions => "azure/audio/transcriptions",
            Self::AudioTranslations => "azure/audio/translations",
            Self::AudioSpeech => "azure/audio/speech",
            Self::ImageGenerations => "azure/images/generations",
            Self::ImageEdits => "azure/images/edits",
        }
    }

    /// Whether the request body carries `model` (set to the deployment
    /// name). The v1 route routes by it, so it is always sent there. On
    /// the deployment route the URL names the deployment; the image
    /// operations drop `model` there, because some image models reject a
    /// deployment name in it.
    pub fn sends_body_model(self, provider_key: &ProviderKey) -> bool {
        uses_v1_route(provider_key) || !matches!(self, Self::ImageGenerations | Self::ImageEdits)
    }
}

/// Whether the key's `api_version` selects the v1 route.
pub fn uses_v1_route(provider_key: &ProviderKey) -> bool {
    dated_api_version(provider_key.api_version.as_deref(), "").is_none()
}

/// The `api-version` a request sends for a configured `api_version`:
/// `None` for `v1` / `preview` / `latest`, which select the v1 route and
/// send none (the three are aliases), else the configured dated value or
/// `default`. For the routes that build their own URL shape (realtime,
/// files / batches / fine-tuning).
pub fn dated_api_version<'a>(api_version: Option<&'a str>, default: &'a str) -> Option<&'a str> {
    match api_version {
        Some("v1" | "preview" | "latest") => None,
        Some(dated) => Some(dated),
        None => Some(default),
    }
}

/// The URL `operation` posts to for `deployment` on this key, cached per
/// (key, operation, deployment).
pub fn endpoint_url(
    provider_key_id: &str,
    provider_key: &ProviderKey,
    deployment: &str,
    operation: AzureOperation,
) -> Result<EndpointUrl, BridgeError> {
    cached_model_endpoint_url(
        provider_key_id,
        operation.url_cache_key(),
        deployment,
        &[
            provider_key.api_base.as_deref().unwrap_or(""),
            provider_key.api_version.as_deref().unwrap_or(""),
        ],
        || build_url(provider_key, deployment, operation),
    )
}

fn build_url(
    provider_key: &ProviderKey,
    deployment: &str,
    operation: AzureOperation,
) -> Result<String, BridgeError> {
    let upstream = AzureUpstreamRef::resolve(deployment, provider_key.api_base.as_deref())?;
    let base = upstream.base_url();
    let path = operation.path();
    if uses_v1_route(provider_key) {
        return Ok(format!("{base}/openai/v1/{path}"));
    }
    let version = provider_key
        .api_version
        .as_deref()
        .unwrap_or(operation.default_api_version());
    Ok(format!(
        "{base}/openai/deployments/{deployment}/{path}?api-version={version}"
    ))
}

/// The auth header for a resolved credential: an API key goes out as
/// `api-key` on every route, an Entra ID token as `Authorization: Bearer`.
pub fn auth_header(credential: &AzureCredential) -> Result<(HeaderName, HeaderValue), BridgeError> {
    match credential {
        AzureCredential::ApiKey(key) => HeaderValue::from_str(key)
            .map(|v| (HeaderName::from_static("api-key"), v))
            .map_err(|_| {
                BridgeError::InvalidUpstreamCredentials(
                    "api key contains invalid header characters".into(),
                )
            }),
        AzureCredential::Bearer(token) => HeaderValue::from_str(&format!("Bearer {token}"))
            .map(|v| (http::header::AUTHORIZATION, v))
            .map_err(|_| {
                BridgeError::Config(
                    "azure aad access_token contains invalid header characters".into(),
                )
            }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pk(api_base: &str, api_version: Option<&str>) -> ProviderKey {
        let mut doc = serde_json::json!({
            "display_name": "az",
            "api_key": "k",
            "api_base": api_base,
            "adapter": "azure-openai",
        });
        if let Some(v) = api_version {
            doc["api_version"] = v.into();
        }
        serde_json::from_value(doc).unwrap()
    }

    #[test]
    fn unset_version_takes_each_operation_default_on_the_deployment_route() {
        let key = pk("https://acme.openai.azure.com", None);
        let cases = [
            (
                AzureOperation::ChatCompletions,
                "chat/completions?api-version=2024-10-21",
            ),
            (
                AzureOperation::Embeddings,
                "embeddings?api-version=2024-10-21",
            ),
            (
                AzureOperation::AudioTranscriptions,
                "audio/transcriptions?api-version=2024-10-21",
            ),
            (
                AzureOperation::AudioTranslations,
                "audio/translations?api-version=2024-10-21",
            ),
            (
                AzureOperation::AudioSpeech,
                "audio/speech?api-version=2025-04-01-preview",
            ),
            (
                AzureOperation::ImageGenerations,
                "images/generations?api-version=2025-04-01-preview",
            ),
            (
                AzureOperation::ImageEdits,
                "images/edits?api-version=2025-04-01-preview",
            ),
        ];
        for (op, tail) in cases {
            assert_eq!(
                build_url(&key, "dep", op).unwrap(),
                format!("https://acme.openai.azure.com/openai/deployments/dep/{tail}")
            );
        }
    }

    #[test]
    fn a_dated_version_overrides_every_default() {
        let key = pk("http://proxy.internal/prefix/", Some("2025-03-01-preview"));
        assert_eq!(
            build_url(&key, "img", AzureOperation::ImageGenerations).unwrap(),
            "http://proxy.internal/prefix/openai/deployments/img/images/generations?api-version=2025-03-01-preview"
        );
        assert_eq!(
            build_url(&key, "chat", AzureOperation::ChatCompletions).unwrap(),
            "http://proxy.internal/prefix/openai/deployments/chat/chat/completions?api-version=2025-03-01-preview"
        );
    }

    #[test]
    fn v1_versions_take_the_v1_route_without_a_query() {
        for v in ["v1", "preview", "latest"] {
            let key = pk("acme", Some(v));
            assert_eq!(
                build_url(&key, "dep", AzureOperation::ImageEdits).unwrap(),
                "https://acme.openai.azure.com/openai/v1/images/edits"
            );
            assert!(AzureOperation::ImageGenerations.sends_body_model(&key));
        }
    }

    #[test]
    fn image_operations_drop_body_model_only_on_the_deployment_route() {
        let key = pk("acme", Some("2024-10-21"));
        assert!(!AzureOperation::ImageGenerations.sends_body_model(&key));
        assert!(!AzureOperation::ImageEdits.sends_body_model(&key));
        assert!(AzureOperation::ChatCompletions.sends_body_model(&key));
        assert!(AzureOperation::AudioSpeech.sends_body_model(&key));
    }

    #[test]
    fn an_api_key_never_goes_out_as_a_bearer() {
        let (name, value) = auth_header(&AzureCredential::ApiKey("k1".into())).unwrap();
        assert_eq!((name.as_str(), value.to_str().unwrap()), ("api-key", "k1"));
        let (name, value) = auth_header(&AzureCredential::Bearer("t1".into())).unwrap();
        assert_eq!(
            (name.as_str(), value.to_str().unwrap()),
            ("authorization", "Bearer t1")
        );
    }

    #[test]
    fn a_changed_api_version_is_not_served_from_the_url_cache() {
        let dated = pk("http://127.0.0.1:9", Some("2024-10-21"));
        let v1 = pk("http://127.0.0.1:9", Some("v1"));
        let url = |key: &ProviderKey| match endpoint_url(
            "pk-cache-test",
            key,
            "dep",
            AzureOperation::ChatCompletions,
        )
        .unwrap()
        {
            EndpointUrl::Parsed(u) => u.to_string(),
            EndpointUrl::Unparsed(s) => s,
        };
        assert!(url(&dated).contains("/deployments/dep/"));
        assert_eq!(url(&v1), "http://127.0.0.1:9/openai/v1/chat/completions");
    }
}
