//! Vertex AI request/response wire shapes.
//!
//! **Skeleton:** only the constants and helpers the bridge needs to
//! sketch the publisher-dispatch path. Real `generateContent` /
//! `streamGenerateContent` request bodies and `streamRawPredict`
//! wrappers (for Anthropic-on-Vertex) land in follow-up D5.x PRs.

/// Query parameters reserved by the Vertex REST API that
/// `default_headers` / `default_body_fields` must never overwrite.
/// Vertex pins the API mode via path *and* via the `alt` query
/// parameter; a misconfigured override block must not redirect SSE
/// streaming to JSON or vice versa.
///
/// Returned by value (not a const slice) to keep the public surface
/// import-free for callers that only need to iterate — the list is
/// tiny so the allocation is negligible.
pub(crate) fn reserved_query_params() -> &'static [&'static str] {
    &["alt", "key", "access_token"]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserved_query_params_covers_vertex_auth_keys() {
        // Vertex accepts `?key=<api-key>` (legacy / Gemini-API path)
        // and `?access_token=<bearer>` (OAuth path) on top of the
        // Authorization header. An override block must not be able
        // to inject either of these.
        let reserved = reserved_query_params();
        assert!(reserved.contains(&"alt"));
        assert!(reserved.contains(&"key"));
        assert!(reserved.contains(&"access_token"));
    }
}
