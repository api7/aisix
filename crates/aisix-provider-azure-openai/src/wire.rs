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
/// `default_headers` must never inject. Azure supports two auth
/// modes:
///
/// - `api-key: <key>` — legacy / RBAC-disabled tenants
/// - `Authorization: Bearer <aad-token>` — Entra (AAD) RBAC
///
/// Reserving both prevents a future AAD-mode operator from poking
/// either header through the override block — same defense-in-depth
/// pattern OpenAiBridge uses for `Authorization` / `x-api-key`.
///
/// Values are lowercase canonical so they compare case-insensitively
/// against `http::HeaderName::as_str()` (which lowercases on parse).
pub(crate) fn reserved_auth_headers() -> &'static [&'static str] {
    &["api-key", "authorization"]
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
    fn reserved_auth_headers_pins_both_azure_auth_modes() {
        // Azure supports `api-key: <key>` AND
        // `Authorization: Bearer <aad-token>` (Entra RBAC). A
        // default_headers block trying to inject EITHER must be
        // dropped at apply time — same defense-in-depth contract
        // OpenAiBridge uses for its Bearer / vendor api-key headers.
        let reserved = reserved_auth_headers();
        assert!(
            reserved.contains(&"api-key"),
            "must reserve api-key (legacy auth)"
        );
        assert!(
            reserved.contains(&"authorization"),
            "must reserve authorization (AAD Bearer auth)"
        );
    }
}
