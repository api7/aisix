macro_rules! provider {
    (@chat_path) => {};
    (@chat_path $path:literal) => {
        fn chat_endpoint_path(&self, _model: &str) -> std::borrow::Cow<'static, str> {
            std::borrow::Cow::Borrowed($path)
        }
    };

    (@stream) => {};
    (@stream $kind:expr) => {
        fn stream_reader_kind(&self) -> $crate::gateway::traits::StreamReaderKind {
            $kind
        }
    };

    (@auth bearer) => {
        fn build_auth_headers(
            &self,
            auth: &$crate::gateway::provider_instance::ProviderAuth,
        ) -> $crate::gateway::error::Result<http::HeaderMap> {
            let mut headers = http::HeaderMap::new();
            let value = http::HeaderValue::from_str(&format!(
                "Bearer {}",
                auth.api_key_for(self.name())?
            ))
                .map_err(|error| $crate::gateway::error::GatewayError::Validation(error.to_string()))?;
            headers.insert(http::header::AUTHORIZATION, value);
            Ok(headers)
        }
    };

    (@auth api_key_header($header:literal)) => {
        fn build_auth_headers(
            &self,
            auth: &$crate::gateway::provider_instance::ProviderAuth,
        ) -> $crate::gateway::error::Result<http::HeaderMap> {
            const HEADER_NAME: http::header::HeaderName =
                http::header::HeaderName::from_static($header);
            let mut headers = http::HeaderMap::new();
            let value = http::HeaderValue::from_str(auth.api_key_for(self.name())?)
                .map_err(|error| $crate::gateway::error::GatewayError::Validation(error.to_string()))?;
            headers.insert(HEADER_NAME, value);
            Ok(headers)
        }
    };

    (@quirks { $($field:ident : $value:expr),* $(,)? }) => {
        fn default_quirks(&self) -> $crate::gateway::traits::CompatQuirks {
            $crate::gateway::traits::CompatQuirks {
                $($field: $value,)*
                ..$crate::gateway::traits::CompatQuirks::NONE
            }
        }
    };

    (@impl_provider
        $name:ident,
        $display:literal,
        $base:literal,
        [$($path:tt)?],
        [$($stream_kind:tt)?],
        [$($auth_kind:tt)+]
    ) => {
        pub struct $name;

        impl $crate::gateway::traits::ProviderMeta for $name {
            fn name(&self) -> &'static str {
                $display
            }

            fn default_base_url(&self) -> &'static str {
                $base
            }

            provider!(@chat_path $($path)?);
            provider!(@stream $($stream_kind)?);
            provider!(@auth $($auth_kind)+);
        }

        impl $crate::gateway::traits::ProviderCapabilities for $name {}
    };

    (
        $name:ident {
            display_name: $display:literal,
            base_url: $base:literal,
            $(chat_path: $path:literal,)?
            $(stream: $stream_kind:expr,)?
            auth: bearer,
            quirks: { $($quirk_field:ident : $quirk_value:expr),* $(,)? }
        }
    ) => {
        provider!(@impl_provider $name, $display, $base, [$($path)?], [$($stream_kind)?], [bearer]);

        impl $crate::gateway::traits::ChatTransform for $name {
            provider!(@quirks { $($quirk_field : $quirk_value),* });
        }
    };

    (
        $name:ident {
            display_name: $display:literal,
            base_url: $base:literal,
            $(chat_path: $path:literal,)?
            $(stream: $stream_kind:expr,)?
            auth: bearer $(,)?
        }
    ) => {
        provider!(@impl_provider $name, $display, $base, [$($path)?], [$($stream_kind)?], [bearer]);

        impl $crate::gateway::traits::ChatTransform for $name {}
    };

    (
        $name:ident {
            display_name: $display:literal,
            base_url: $base:literal,
            $(chat_path: $path:literal,)?
            $(stream: $stream_kind:expr,)?
            auth: api_key_header($header:literal),
            quirks: { $($quirk_field:ident : $quirk_value:expr),* $(,)? }
        }
    ) => {
        provider!(@impl_provider $name, $display, $base, [$($path)?], [$($stream_kind)?], [api_key_header($header)]);

        impl $crate::gateway::traits::ChatTransform for $name {
            provider!(@quirks { $($quirk_field : $quirk_value),* });
        }
    };

    (
        $name:ident {
            display_name: $display:literal,
            base_url: $base:literal,
            $(chat_path: $path:literal,)?
            $(stream: $stream_kind:expr,)?
            auth: api_key_header($header:literal) $(,)?
        }
    ) => {
        provider!(@impl_provider $name, $display, $base, [$($path)?], [$($stream_kind)?], [api_key_header($header)]);

        impl $crate::gateway::traits::ChatTransform for $name {}
    };
}

pub(crate) use provider;

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use crate::gateway::{
        provider_instance::ProviderAuth,
        traits::{ChatTransform, ProviderMeta, StreamReaderKind},
    };

    provider!(MacroTestProvider {
        display_name: "macro-test",
        base_url: "https://provider.example.com",
        chat_path: "/custom/chat",
        stream: StreamReaderKind::JsonArrayStream,
        auth: api_key_header("x-api-key"),
        quirks: {
            unsupported_params: &["seed"],
            tool_args_may_be_object: true,
        }
    });

    #[test]
    fn macro_generated_provider_exposes_expected_metadata() {
        let provider = MacroTestProvider;

        assert_eq!(provider.name(), "macro-test");
        assert_eq!(provider.default_base_url(), "https://provider.example.com");
        assert_eq!(
            provider.chat_endpoint_path("ignored"),
            Cow::Borrowed("/custom/chat")
        );
        assert_eq!(
            provider.stream_reader_kind(),
            StreamReaderKind::JsonArrayStream
        );
    }

    #[test]
    fn macro_generated_provider_builds_auth_headers_and_quirks() {
        let provider = MacroTestProvider;
        let headers = provider
            .build_auth_headers(&ProviderAuth::ApiKey("secret-key".into()))
            .unwrap();
        let quirks = provider.default_quirks();

        assert_eq!(headers["x-api-key"], "secret-key");
        assert_eq!(quirks.unsupported_params, &["seed"]);
        assert!(quirks.tool_args_may_be_object);
    }

    #[test]
    fn macro_generated_provider_reports_provider_name_for_missing_api_key() {
        let provider = MacroTestProvider;
        let error = provider
            .build_auth_headers(&ProviderAuth::None)
            .unwrap_err();

        assert!(matches!(
            error,
            crate::gateway::error::GatewayError::Validation(message)
                if message.contains("macro-test")
                    && message.contains("ProviderAuth::ApiKey")
        ));
    }
}
