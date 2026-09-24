//! Redaction of credentials embedded in configured upstream URLs.

use std::borrow::Cow;

/// `url` with any userinfo replaced by `***`, for rendering a configured
/// upstream URL into a log line, a `Debug` impl or an error message.
///
/// Several upstream URLs legitimately carry `user:pass@` — it is how a
/// Basic-auth upstream is configured where the resource has no other
/// slot for it, and the HTTP client turns it into an `Authorization`
/// header — so the gateway accepts it and keeps it out of everything it
/// prints instead. Every place that renders such a URL goes through this
/// one function, so the rendered form cannot drift between them.
///
/// Scheme, host, port, path and query are kept for diagnosis. Only the
/// authority is inspected: an `@` in the path is not userinfo. A value
/// with no scheme is treated as a bare authority. The last `@` of the
/// authority ends the userinfo, so a password with an unencoded `@` is
/// redacted whole.
pub fn redact_url_userinfo(url: &str) -> Cow<'_, str> {
    let (prefix, rest) = match url.split_once("://") {
        Some((scheme, rest)) => (&url[..scheme.len() + 3], rest),
        None => ("", url),
    };
    let authority_end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    let (authority, tail) = rest.split_at(authority_end);
    match authority.rfind('@') {
        Some(at) => Cow::Owned(format!("{prefix}***{}{tail}", &authority[at..])),
        None => Cow::Borrowed(url),
    }
}

#[cfg(test)]
mod tests {
    use super::redact_url_userinfo;

    #[test]
    fn replaces_userinfo_and_keeps_the_rest() {
        for (input, expected) in [
            (
                "https://user:secret@proxy.internal",
                "https://***@proxy.internal",
            ),
            (
                "http://just-user@host:8001/path?q=1",
                "http://***@host:8001/path?q=1",
            ),
            ("https://u:p@ss@host/x", "https://***@host/x"),
            ("user:pw@etcd.example.com:7943", "***@etcd.example.com:7943"),
        ] {
            assert_eq!(redact_url_userinfo(input), expected);
        }
    }

    #[test]
    fn leaves_urls_without_userinfo_untouched() {
        for input in [
            "https://proxy.internal/x",
            "https://proxy.internal/v1@my-namespace",
            "https://proxy.internal?who=a@b",
            "",
        ] {
            assert_eq!(redact_url_userinfo(input), input);
        }
    }
}
