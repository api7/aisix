use http::HeaderMap;
use reqwest::Url;

use crate::{
    config::entities::{Model, ResourceEntry, models::ProviderConfig},
    gateway::{
        Gateway,
        error::{GatewayError, Result},
        provider_instance::{ProviderAuth, ProviderInstance},
    },
    providers::{Provider, create_provider},
};

pub fn create_legacy_provider(model: &ResourceEntry<Model>) -> Box<dyn Provider> {
    create_provider(&model.provider_config)
}

pub fn create_provider_instance(
    gateway: &Gateway,
    model: &ResourceEntry<Model>,
) -> Result<ProviderInstance> {
    let provider_name = model.model.provider.as_str();
    let def = gateway.registry().get(provider_name).ok_or_else(|| {
        GatewayError::Validation(format!(
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
        Some(api_base) => Some(Url::parse(api_base).map_err(|error| {
            GatewayError::Validation(format!("invalid provider api_base {}: {}", api_base, error))
        })?),
        None => None,
    };

    Ok((ProviderAuth::ApiKey(api_key.clone()), base_url_override))
}
