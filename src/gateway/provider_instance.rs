use std::{collections::HashMap, fmt, sync::Arc};

use http::HeaderMap;
use reqwest::Url;

use crate::gateway::{
    error::{GatewayError, Result},
    traits::ProviderCapabilities,
};

/// Authentication material bound to a provider instance at runtime.
#[derive(Clone, Default)]
pub enum ProviderAuth {
    ApiKey(String),
    #[default]
    None,
}

impl fmt::Debug for ProviderAuth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ApiKey(_) => f.write_str("ApiKey(REDACTED)"),
            Self::None => f.write_str("None"),
        }
    }
}

impl ProviderAuth {
    pub fn api_key(&self) -> Result<&str> {
        match self {
            Self::ApiKey(api_key) => Ok(api_key),
            Self::None => Err(GatewayError::Validation(
                "provider requires api key authentication".into(),
            )),
        }
    }
}

/// Runtime provider configuration: definition, auth, and deployment overrides.
#[derive(Clone)]
pub struct ProviderInstance {
    pub def: Arc<dyn ProviderCapabilities>,
    pub auth: ProviderAuth,
    pub base_url_override: Option<Url>,
    pub custom_headers: HeaderMap,
}

impl ProviderInstance {
    pub fn effective_base_url(&self) -> Result<Url> {
        if let Some(base_url) = &self.base_url_override {
            return Ok(base_url.clone());
        }

        self.def.default_base_url().parse().map_err(|error| {
            GatewayError::Validation(format!(
                "provider {} has invalid default_base_url {}: {}",
                self.def.name(),
                self.def.default_base_url(),
                error
            ))
        })
    }

    pub fn build_url(&self, model: &str) -> Result<String> {
        let base_url = self.effective_base_url()?;
        Ok(self.def.build_url(base_url.as_str(), model))
    }

    pub fn build_headers(&self) -> Result<HeaderMap> {
        let mut headers = self.def.build_auth_headers(&self.auth)?;
        headers.extend(self.custom_headers.clone());
        Ok(headers)
    }
}

/// Immutable registry of provider definitions.
pub struct ProviderRegistry {
    defs: HashMap<&'static str, Arc<dyn ProviderCapabilities>>,
}

impl ProviderRegistry {
    pub fn builder() -> ProviderRegistryBuilder {
        ProviderRegistryBuilder {
            defs: HashMap::new(),
        }
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn ProviderCapabilities>> {
        self.defs.get(name).cloned()
    }
}

pub struct ProviderRegistryBuilder {
    defs: HashMap<&'static str, Arc<dyn ProviderCapabilities>>,
}

impl ProviderRegistryBuilder {
    pub fn register<P: ProviderCapabilities + 'static>(mut self, provider: P) -> Result<Self> {
        if self.defs.contains_key(provider.name()) {
            return Err(GatewayError::Validation(format!(
                "provider {} is already registered",
                provider.name()
            )));
        }

        self.defs.insert(provider.name(), Arc::new(provider));
        Ok(self)
    }

    pub fn build(self) -> ProviderRegistry {
        ProviderRegistry { defs: self.defs }
    }
}

#[cfg(test)]
mod tests {
    use std::{borrow::Cow, sync::Arc};

    use http::{
        HeaderMap, HeaderValue,
        header::{AUTHORIZATION, HeaderName},
    };

    use super::{ProviderAuth, ProviderInstance, ProviderRegistry};
    use crate::gateway::{
        error::{GatewayError, Result},
        traits::{ChatTransform, ProviderCapabilities, ProviderMeta, StreamReaderKind},
    };

    struct DummyProvider;

    struct InvalidUrlProvider;

    struct DuplicateDummyProvider;

    impl ProviderMeta for DummyProvider {
        fn name(&self) -> &'static str {
            "dummy"
        }

        fn default_base_url(&self) -> &'static str {
            "https://api.example.com/"
        }

        fn chat_endpoint_path(&self, model: &str) -> Cow<'static, str> {
            Cow::Owned(format!("/v1/models/{model}/chat"))
        }

        fn stream_reader_kind(&self) -> StreamReaderKind {
            StreamReaderKind::Sse
        }

        fn build_auth_headers(&self, auth: &ProviderAuth) -> Result<HeaderMap> {
            let mut headers = HeaderMap::new();
            if let ProviderAuth::ApiKey(api_key) = auth {
                let value = HeaderValue::from_str(&format!("Bearer {api_key}"))
                    .map_err(|error| GatewayError::Validation(error.to_string()))?;
                headers.insert(AUTHORIZATION, value);
            }
            Ok(headers)
        }
    }

    impl ChatTransform for DummyProvider {}

    impl ProviderCapabilities for DummyProvider {}

    impl ProviderMeta for InvalidUrlProvider {
        fn name(&self) -> &'static str {
            "invalid-url"
        }

        fn default_base_url(&self) -> &'static str {
            "not a url"
        }

        fn stream_reader_kind(&self) -> StreamReaderKind {
            StreamReaderKind::Sse
        }

        fn build_auth_headers(&self, _auth: &ProviderAuth) -> Result<HeaderMap> {
            Ok(HeaderMap::new())
        }
    }

    impl ChatTransform for InvalidUrlProvider {}

    impl ProviderCapabilities for InvalidUrlProvider {}

    impl ProviderMeta for DuplicateDummyProvider {
        fn name(&self) -> &'static str {
            "dummy"
        }

        fn default_base_url(&self) -> &'static str {
            "https://duplicate.example.com"
        }

        fn stream_reader_kind(&self) -> StreamReaderKind {
            StreamReaderKind::Sse
        }

        fn build_auth_headers(&self, _auth: &ProviderAuth) -> Result<HeaderMap> {
            Ok(HeaderMap::new())
        }
    }

    impl ChatTransform for DuplicateDummyProvider {}

    impl ProviderCapabilities for DuplicateDummyProvider {}

    #[test]
    fn provider_auth_debug_redacts_api_key() {
        assert_eq!(
            format!("{:?}", ProviderAuth::ApiKey("sk-secret".into())),
            "ApiKey(REDACTED)"
        );
        assert_eq!(format!("{:?}", ProviderAuth::None), "None");
    }

    #[test]
    fn provider_auth_api_key_requires_api_key_variant() {
        assert_eq!(
            ProviderAuth::ApiKey("sk-secret".into()).api_key().unwrap(),
            "sk-secret"
        );

        let error = ProviderAuth::None.api_key().unwrap_err();
        assert!(matches!(
            error,
            GatewayError::Validation(message)
                if message.contains("api key authentication")
        ));
    }

    #[test]
    fn provider_instance_build_url_uses_provider_path() {
        let instance = ProviderInstance {
            def: Arc::new(DummyProvider),
            auth: ProviderAuth::None,
            base_url_override: None,
            custom_headers: HeaderMap::new(),
        };

        assert_eq!(
            instance.build_url("demo-model").unwrap(),
            "https://api.example.com/v1/models/demo-model/chat"
        );
    }

    #[test]
    fn provider_instance_invalid_default_base_url_returns_validation_error() {
        let instance = ProviderInstance {
            def: Arc::new(InvalidUrlProvider),
            auth: ProviderAuth::None,
            base_url_override: None,
            custom_headers: HeaderMap::new(),
        };

        let error = instance.effective_base_url().unwrap_err();

        assert!(matches!(
            error,
            GatewayError::Validation(message)
                if message.contains("invalid-url")
                    && message.contains("default_base_url")
        ));
    }

    #[test]
    fn provider_instance_build_headers_merges_auth_and_custom_headers() {
        let mut custom_headers = HeaderMap::new();
        custom_headers.insert(
            HeaderName::from_static("x-trace-id"),
            HeaderValue::from_static("trace-123"),
        );
        let instance = ProviderInstance {
            def: Arc::new(DummyProvider),
            auth: ProviderAuth::ApiKey("sk-secret".into()),
            base_url_override: None,
            custom_headers,
        };

        let headers = instance.build_headers().unwrap();

        assert_eq!(headers[AUTHORIZATION], "Bearer sk-secret");
        assert_eq!(headers["x-trace-id"], "trace-123");
    }

    #[test]
    fn provider_registry_registers_and_looks_up_definitions() {
        let registry = ProviderRegistry::builder()
            .register(DummyProvider)
            .unwrap()
            .build();

        let provider = registry.get("dummy").unwrap();
        assert_eq!(provider.name(), "dummy");
        assert!(registry.get("missing").is_none());
    }

    #[test]
    fn provider_registry_rejects_duplicate_names() {
        let error = ProviderRegistry::builder()
            .register(DummyProvider)
            .unwrap()
            .register(DuplicateDummyProvider)
            .err()
            .unwrap();

        assert!(matches!(
            error,
            GatewayError::Validation(message)
                if message.contains("dummy")
                    && message.contains("already registered")
        ));
    }
}
