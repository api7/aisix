//! Delivering a verified caller JWT to an upstream.
//!
//! Three resources carry a `forward_jwt_header` field — `provider_key`'s
//! request overrides, `passthrough_route`, and `mcp_server` — because the
//! gateway fronts internal upstreams on three different surfaces and an
//! operator enables the behaviour per upstream, never gateway-wide: the
//! same deployment reaches internal services that authorize on the end
//! user's claims AND public model providers that must never see a
//! corporate token.
//!
//! The rendering rule lives here rather than at each of the three call
//! sites so all three answer "which header, and in what form" the same
//! way — the drift this repo's handler-family rule warns about.

/// Header names `forward_jwt_header` may never take.
///
/// Every entry describes the MESSAGE — its framing, its body, or the
/// connection carrying it — rather than who sent it, so a token placed
/// there corrupts the request instead of identifying the caller. The
/// credential slots (`authorization`, `x-api-key`, and friends) are
/// deliberately absent: delivering the caller's token into one of them is
/// the whole point of the field, and is what lets an existing internal
/// service keep reading `Authorization` with no change at all.
///
/// The fields' schemars pattern forces lowercase, so this lowercase list
/// is exhaustive on every configuration path (header matching is
/// case-insensitive on the wire regardless).
pub const TRANSPORT_HEADER_SLOTS: [&str; 13] = [
    "accept",
    "accept-encoding",
    "connection",
    "content-encoding",
    "content-length",
    "content-type",
    "expect",
    "host",
    "keep-alive",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
];

/// Headers defined to carry `<scheme> <credentials>` (RFC 9110 §11.6.2),
/// where a bare token is malformed.
const SCHEME_BEARING_SLOTS: [&str; 2] = ["authorization", "proxy-authorization"];

/// The header name and value a verified caller JWT is delivered under, or
/// `None` when nothing should be sent.
///
/// `None` covers the three ordinary ways this does not apply, none of them
/// an error: the upstream has no `forward_jwt_header` configured, the
/// caller authenticated with an API key so there is no token to relay, or
/// the configured name describes the message rather than its sender (the
/// schema rejects those at write time; this is the runtime half of that
/// pair, so a document written by an older control plane cannot corrupt a
/// request).
///
/// The value is the token exactly as verified — no claim added, removed,
/// or rewritten — prefixed with `Bearer ` only for the headers whose own
/// definition requires a scheme.
pub fn forwarded_jwt(configured: Option<&str>, token: Option<&str>) -> Option<(String, String)> {
    let name = configured?.to_ascii_lowercase();
    let token = token?;
    if token.is_empty() || TRANSPORT_HEADER_SLOTS.contains(&name.as_str()) {
        return None;
    }
    let value = if SCHEME_BEARING_SLOTS.contains(&name.as_str()) {
        format!("Bearer {token}")
    } else {
        token.to_string()
    };
    Some((name, value))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unset_field_or_keyed_caller_sends_nothing() {
        assert_eq!(forwarded_jwt(None, Some("tok")), None);
        assert_eq!(forwarded_jwt(Some("x-user-jwt"), None), None);
        assert_eq!(forwarded_jwt(None, None), None);
        // An empty token would put a blank credential on the wire, which
        // an upstream reads as "present but unauthenticated".
        assert_eq!(forwarded_jwt(Some("x-user-jwt"), Some("")), None);
    }

    #[test]
    fn custom_header_carries_the_bare_token() {
        assert_eq!(
            forwarded_jwt(Some("x-user-jwt"), Some("eyJhbGciOi")),
            Some(("x-user-jwt".to_string(), "eyJhbGciOi".to_string()))
        );
    }

    #[test]
    fn scheme_bearing_headers_carry_the_bearer_form() {
        assert_eq!(
            forwarded_jwt(Some("authorization"), Some("eyJhbGciOi")),
            Some(("authorization".to_string(), "Bearer eyJhbGciOi".to_string()))
        );
        assert_eq!(
            forwarded_jwt(Some("proxy-authorization"), Some("t")),
            Some(("proxy-authorization".to_string(), "Bearer t".to_string()))
        );
    }

    #[test]
    fn credential_slots_are_allowed_targets() {
        // The point of the field: an upstream already reading `x-api-key`
        // keeps reading it, with the end user's token in it.
        assert_eq!(
            forwarded_jwt(Some("x-api-key"), Some("t")),
            Some(("x-api-key".to_string(), "t".to_string()))
        );
    }

    #[test]
    fn transport_slots_send_nothing() {
        for name in TRANSPORT_HEADER_SLOTS {
            assert_eq!(forwarded_jwt(Some(name), Some("t")), None, "{name}");
        }
    }

    #[test]
    fn a_mixed_case_name_is_matched_against_the_reject_list() {
        // The schema forces lowercase, but a document written before that
        // constraint — or by a future control plane — must not slip a
        // transport slot through on capitalisation alone.
        assert_eq!(forwarded_jwt(Some("Host"), Some("t")), None);
        assert_eq!(
            forwarded_jwt(Some("Authorization"), Some("t")),
            Some(("authorization".to_string(), "Bearer t".to_string()))
        );
    }
}
