use http::HeaderMap;
use reqwest::Url;

use crate::{
    config::entities::{
        Model, Provider, ResourceEntry, ResourceRegistry, providers::ProviderConfig,
    },
    gateway::{
        Gateway,
        error::{GatewayError, Result},
        provider_instance::{AwsStaticCredentials, ProviderAuth, ProviderInstance},
    },
};

/// Creates a gateway provider instance for the given model using the gateway registry.
#[fastrace::trace]
pub fn create_provider_instance(
    gateway: &Gateway,
    resources: &ResourceRegistry,
    model: &ResourceEntry<Model>,
) -> Result<ProviderInstance> {
    let provider = resolve_provider(resources, &model.provider_id)?;
    let provider_name = provider.provider_type();
    let def = gateway.registry().get(provider_name).ok_or_else(|| {
        GatewayError::Internal(format!(
            "provider {} is not registered in gateway registry",
            provider_name
        ))
    })?;

    let (auth, base_url_override) = provider_auth_and_base_url(&provider.provider)?;

    Ok(ProviderInstance {
        def,
        auth,
        base_url_override,
        custom_headers: HeaderMap::new(),
    })
}

fn resolve_provider(
    resources: &ResourceRegistry,
    provider_id: &str,
) -> Result<ResourceEntry<Provider>> {
    resources
        .providers
        .list()
        .get(provider_id)
        .cloned()
        .ok_or_else(|| GatewayError::Internal(format!("provider {} not found", provider_id)))
}

fn provider_auth_and_base_url(config: &ProviderConfig) -> Result<(ProviderAuth, Option<Url>)> {
    let (auth, base_url_override) = match config {
        ProviderConfig::Anthropic(config) => (
            ProviderAuth::ApiKey(config.api_key.clone()),
            parse_base_url(config.api_base.as_deref())?,
        ),
        ProviderConfig::Bedrock(config) => bedrock_auth_and_base_url(config)?,
        ProviderConfig::DeepSeek(config) => (
            ProviderAuth::ApiKey(config.api_key.clone()),
            parse_base_url(config.api_base.as_deref())?,
        ),
        ProviderConfig::Gemini(config) => (
            ProviderAuth::ApiKey(config.api_key.clone()),
            parse_base_url(config.api_base.as_deref())?,
        ),
        ProviderConfig::OpenAI(config) => (
            ProviderAuth::ApiKey(config.api_key.clone()),
            parse_base_url(config.api_base.as_deref())?,
        ),
    };

    Ok((auth, base_url_override))
}

fn bedrock_auth_and_base_url(
    config: &crate::gateway::providers::configs::BedrockProviderConfig,
) -> Result<(ProviderAuth, Option<Url>)> {
    let auth = ProviderAuth::AwsStatic(AwsStaticCredentials {
        access_key_id: config.access_key_id.clone(),
        secret_access_key: config.secret_access_key.clone(),
        session_token: config.session_token.clone(),
        region: config.region.clone(),
    });
    let base_url_override = match config.endpoint.as_deref() {
        Some(endpoint) => parse_base_url(Some(endpoint))?,
        None => parse_base_url(Some(&default_bedrock_base_url(config.region.as_str())))?,
    };

    Ok((auth, base_url_override))
}

fn parse_base_url(api_base: Option<&str>) -> Result<Option<Url>> {
    match api_base {
        Some(api_base) => {
            let parsed = Url::parse(api_base).map_err(|error| {
                GatewayError::Internal(format!("invalid provider api_base {}: {}", api_base, error))
            })?;

            if !matches!(parsed.scheme(), "http" | "https") {
                return Err(GatewayError::Internal(format!(
                    "invalid provider api_base {}: unsupported scheme {}",
                    api_base,
                    parsed.scheme()
                )));
            }

            Ok(Some(parsed))
        }
        None => Ok(None),
    }
}

fn default_bedrock_base_url(region: &str) -> String {
    format!("https://bedrock-runtime.{region}.amazonaws.com")
}

#[cfg(test)]
mod tests {
    use reqwest::Url;

    use super::provider_auth_and_base_url;
    use crate::{
        config::entities::providers::ProviderConfig,
        gateway::providers::configs::BedrockProviderConfig,
    };

    #[test]
    fn provider_auth_and_base_url_returns_bedrock_static_credentials() {
        let config = ProviderConfig::Bedrock(BedrockProviderConfig {
            region: "us-east-1".into(),
            access_key_id: "AKIA123".into(),
            secret_access_key: "secret".into(),
            session_token: Some("token".into()),
            endpoint: Some("https://bedrock-runtime.us-east-1.amazonaws.com".into()),
        });

        let (auth, base_url_override) = provider_auth_and_base_url(&config).unwrap();
        let credentials = auth.aws_static_credentials_for("bedrock").unwrap();

        assert_eq!(credentials.access_key_id, "AKIA123");
        assert_eq!(credentials.secret_access_key, "secret");
        assert_eq!(credentials.session_token.as_deref(), Some("token"));
        assert_eq!(credentials.region, "us-east-1");
        assert_eq!(
            base_url_override.as_ref().map(Url::as_str),
            Some("https://bedrock-runtime.us-east-1.amazonaws.com/")
        );
    }

    #[test]
    fn provider_auth_and_base_url_derives_bedrock_runtime_endpoint_from_region() {
        let config = ProviderConfig::Bedrock(BedrockProviderConfig {
            region: "ap-southeast-1".into(),
            access_key_id: "AKIA123".into(),
            secret_access_key: "secret".into(),
            session_token: None,
            endpoint: None,
        });

        let (auth, base_url_override) = provider_auth_and_base_url(&config).unwrap();
        let credentials = auth.aws_static_credentials_for("bedrock").unwrap();

        assert_eq!(credentials.region, "ap-southeast-1");
        assert_eq!(
            base_url_override.as_ref().map(Url::as_str),
            Some("https://bedrock-runtime.ap-southeast-1.amazonaws.com/")
        );
    }

    #[test]
    fn provider_auth_and_base_url_rejects_invalid_bedrock_endpoint_scheme() {
        let config = ProviderConfig::Bedrock(BedrockProviderConfig {
            region: "us-east-1".into(),
            access_key_id: "AKIA123".into(),
            secret_access_key: "secret".into(),
            session_token: None,
            endpoint: Some("ftp://bedrock-runtime.us-east-1.amazonaws.com".into()),
        });

        let error = provider_auth_and_base_url(&config).unwrap_err();

        assert!(matches!(
            error,
            crate::gateway::error::GatewayError::Internal(message)
                if message.contains("unsupported scheme")
        ));
    }
}
