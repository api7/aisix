use http::HeaderMap;
use reqwest::Url;

use crate::{
    config::entities::{Model, ResourceEntry, models::ProviderConfig},
    gateway::{
        Gateway,
        error::{GatewayError, Result},
        provider_instance::{ProviderAuth, ProviderInstance},
    },
};

/// Creates a gateway provider instance for the given model using the gateway registry.
#[fastrace::trace]
pub fn create_provider_instance(
    gateway: &Gateway,
    model: &ResourceEntry<Model>,
) -> Result<ProviderInstance> {
    let provider_name = model.model.provider.as_str();
    let def = gateway.registry().get(provider_name).ok_or_else(|| {
        GatewayError::Internal(format!(
            "provider {} is not registered in gateway registry",
            provider_name
        ))
    })?;

    let (auth, base_url_override) = provider_auth_and_base_url(&model.provider_config)?;

    Ok(ProviderInstance {
        def,
        auth,
        base_url_override,
        custom_headers: HeaderMap::new(),
    })
}

fn provider_auth_and_base_url(config: &ProviderConfig) -> Result<(ProviderAuth, Option<Url>)> {
    let (api_key, api_base) = match config {
        ProviderConfig::Anthropic(config) => (&config.api_key, config.api_base.as_deref()),
        ProviderConfig::DeepSeek(config) => (&config.api_key, config.api_base.as_deref()),
        ProviderConfig::Gemini(config) => (&config.api_key, config.api_base.as_deref()),
        ProviderConfig::OpenAI(config) => (&config.api_key, config.api_base.as_deref()),
    };

    let base_url_override = match api_base {
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

            Some(parsed)
        }
        None => None,
    };

    Ok((ProviderAuth::ApiKey(api_key.clone()), base_url_override))
}
