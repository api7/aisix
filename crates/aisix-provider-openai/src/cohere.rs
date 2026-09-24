//! Where a Cohere Provider Key's requests go.
//!
//! Chat and embeddings use Cohere's OpenAI-compatible surface, which lives
//! under `/compatibility/v1` (`<host>/v1/chat/completions` answers 405 and
//! `<host>/v1/embeddings` 404). Rerank has no compatible counterpart there
//! and uses the native `<host>/v2/rerank`. The control plane writes a
//! Cohere `api_base` in either of two forms — the bare host, or the host
//! with `/compatibility/v1` — so every route derives its URL from the host
//! root rather than appending to `api_base` as written.
//!
//! <https://docs.cohere.com/docs/compatibility-api>,
//! <https://docs.cohere.com/reference/rerank>

/// Path of Cohere's OpenAI-compatible surface under the host root.
pub const COMPATIBILITY_PATH: &str = "/compatibility/v1";

/// Whether `provider` (a Provider Key or Model vendor id) names Cohere.
pub fn is_cohere(provider: &str) -> bool {
    provider.trim().eq_ignore_ascii_case("cohere")
}

/// The host root of a Cohere `api_base`: trailing slashes and the
/// `/compatibility/v1` suffix removed.
pub fn api_root(base: &str) -> &str {
    let trimmed = base.trim().trim_end_matches('/');
    trimmed
        .strip_suffix(COMPATIBILITY_PATH)
        .unwrap_or(trimmed)
        .trim_end_matches('/')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn api_root_accepts_both_control_plane_forms() {
        for base in [
            "https://api.cohere.com",
            "https://api.cohere.com/",
            "https://api.cohere.com/compatibility/v1",
            "https://api.cohere.com/compatibility/v1/",
            " https://api.cohere.com/compatibility/v1 ",
        ] {
            assert_eq!(api_root(base), "https://api.cohere.com", "{base:?}");
        }
        assert_eq!(
            api_root("http://127.0.0.1:9000/proxy/compatibility/v1"),
            "http://127.0.0.1:9000/proxy"
        );
    }
}
