//! Azure OpenAI Service request/response wire shapes.
//!
//! **Skeleton:** only the constants and helpers the bridge needs to
//! sketch the deployment-dispatch path. Real chat-completions request
//! / response wrappers (with Azure-injected
//! `prompt_filter_results` / `content_filter_results` blocks) land
//! in follow-up D6.x PRs.

/// Query parameters reserved by Azure's REST API that
/// `default_headers` / `default_body_fields` must never overwrite.
/// Azure pins the API version via `api-version` query parameter; a
/// misconfigured override block must not redirect calls to a
/// deprecated or unsupported API version.
pub(crate) fn reserved_query_params() -> &'static [&'static str] {
    &["api-version"]
}

/// Header names reserved by Azure OpenAI authentication that
/// `default_headers` must never inject. Azure uses `api-key`
/// (different from OpenAI's `Authorization: Bearer`); the bridge's
/// own auth header must always win.
pub(crate) fn reserved_auth_headers() -> &'static [&'static str] {
    &["api-key"]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserved_query_params_pins_api_version() {
        let reserved = reserved_query_params();
        assert!(reserved.contains(&"api-version"));
    }

    #[test]
    fn reserved_auth_headers_pins_azure_api_key() {
        // The `api-key` header name is Azure's auth convention.
        // A default_headers block trying to inject it must be
        // dropped at apply time — same defense-in-depth contract
        // OpenAiBridge uses for `Authorization` / `x-api-key`.
        let reserved = reserved_auth_headers();
        assert!(reserved.contains(&"api-key"));
    }
}
