use crate::gateway::providers::macros::provider;

provider!(GoogleDef {
    display_name: "gemini",
    base_url: "https://generativelanguage.googleapis.com/v1beta/openai",
    chat_path: "/chat/completions",
    auth: api_key_header("x-goog-api-key"),
});

#[cfg(test)]
mod tests {
    use super::GoogleDef;
    use crate::gateway::{provider_instance::ProviderAuth, traits::ProviderMeta};

    #[test]
    fn google_def_uses_compatible_gemini_endpoint_and_auth_header() {
        let provider = GoogleDef;
        let headers = provider
            .build_auth_headers(&ProviderAuth::ApiKey("gemini-key".into()))
            .unwrap();

        assert_eq!(provider.name(), "gemini");
        assert_eq!(
            provider.default_base_url(),
            "https://generativelanguage.googleapis.com/v1beta/openai"
        );
        assert_eq!(
            provider.build_url(provider.default_base_url(), "gemini-2.5-flash"),
            "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        );
        assert_eq!(headers["x-goog-api-key"], "gemini-key");
    }
}
