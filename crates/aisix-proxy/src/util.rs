//! Small shared helpers for the proxy layer.

/// Truncate `s` to at most `max_bytes` bytes, rounding down to a UTF-8
/// char boundary, and append an ellipsis when truncation occurred.
///
/// Slicing a `String`/`str` at a fixed byte offset (`&s[..n]`) panics
/// when the offset splits a multibyte codepoint. A non-ASCII upstream
/// error body can trigger that on the error-handling path (issue #420),
/// so every site that truncates an upstream error body for a log line
/// or error envelope must go through this helper rather than a raw
/// byte slice.
pub(crate) fn truncate_on_char_boundary(s: &str, max_bytes: usize) -> String {
    if s.len() <= max_bytes {
        return s.to_string();
    }
    let end = (0..=max_bytes)
        .rev()
        .find(|&i| s.is_char_boundary(i))
        .unwrap_or(0);
    format!("{}…", &s[..end])
}

/// Wire bytes of a request body a proxy route sends upstream itself,
/// bypassing the provider bridges. Build the outbound `Value` inside a block
/// that ends here, so only these bytes — which reqwest holds until the
/// response head — outlive it; see [`aisix_gateway::json_body`]. A reader
/// that needs the outbound form after the upstream answered uses
/// [`outbound_text`] / [`outbound_value`] instead of a kept copy.
pub(crate) fn outbound_json(
    body: &serde_json::Value,
) -> Result<bytes::Bytes, crate::error::ProxyError> {
    aisix_gateway::json_body(body).map_err(|e| {
        crate::error::ProxyError::Bridge(aisix_gateway::BridgeError::Config(format!(
            "serialize request body: {e}"
        )))
    })
}

/// The outbound request as text, for content capture: the wire bytes are
/// exactly what `serde_json::to_string` of the body would produce.
pub(crate) fn outbound_text(outbound: &[u8]) -> String {
    String::from_utf8_lossy(outbound).into_owned()
}

/// The outbound request as a `Value`, rebuilt from its wire bytes for a
/// reader that runs after the upstream has answered.
pub(crate) fn outbound_value(outbound: &[u8]) -> serde_json::Value {
    serde_json::from_slice(outbound).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_string_is_returned_unchanged() {
        assert_eq!(truncate_on_char_boundary("hello", 1024), "hello");
    }

    #[test]
    fn ascii_over_limit_truncates_and_appends_ellipsis() {
        let s = "a".repeat(2000);
        assert_eq!(
            truncate_on_char_boundary(&s, 1024),
            format!("{}…", "a".repeat(1024))
        );
    }

    #[test]
    fn multibyte_body_does_not_panic_and_rounds_down_to_boundary() {
        // '€' is 3 bytes; 1024 is not a multiple of 3, so a raw
        // `&s[..1024]` slice would split a codepoint and panic. The
        // helper must round down to byte 1023 (341 whole '€').
        let s = "€".repeat(1000); // 3000 bytes
        let out = truncate_on_char_boundary(&s, 1024);
        let kept = out.strip_suffix('…').expect("ellipsis appended");
        assert_eq!(kept, "€".repeat(341)); // 341 * 3 = 1023 bytes
        assert!(kept.len() <= 1024);
    }
}
