//! AWS Bedrock Runtime request/response wire shapes.
//!
//! **Skeleton:** only the constants the bridge needs to sketch the
//! per-publisher dispatch path. Real per-publisher request bodies
//! (Anthropic Messages with `anthropic_version` in body, Titan's
//! `inputText`, Llama's `prompt`, Converse for Nova, etc.) plus AWS
//! event-stream framing for streaming land in follow-up D7.x PRs.

/// Header names AWS SigV4 v4 signing owns. These headers MUST NOT be
/// settable via `default_headers` — the bridge's SigV4 signer
/// computes them per-request from the canonical request, and an
/// operator-supplied override would invalidate the signature.
///
/// Same defense-in-depth pattern as OpenAiBridge's
/// `RESERVED_DEFAULT_HEADERS` — cp-api should reject these at write
/// time, but the DP enforces it again at apply time.
pub(crate) fn reserved_sigv4_headers() -> &'static [&'static str] {
    &[
        // SigV4 canonical headers — the signer computes / derives all of
        // these. Any operator override would either get overwritten
        // (best case) or break the signature (worst case).
        "authorization",
        "x-amz-date",
        "x-amz-content-sha256",
        "x-amz-security-token",
        "x-amz-target",
        // Bedrock-specific: the response-stream content-type is a
        // wire-shape gate (binary event-stream vs JSON). An override
        // injecting the wrong value would break stream parsing.
        "x-amzn-bedrock-accept",
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserved_sigv4_headers_covers_signature_inputs() {
        // Tight pin on the SigV4 reserved-header list. A future
        // dispatch PR that adds `x-amz-content-sha256` to the signed
        // canonical headers must NOT also let `default_headers`
        // overwrite it — that would silently break signing.
        let reserved = reserved_sigv4_headers();
        assert!(reserved.contains(&"authorization"));
        assert!(reserved.contains(&"x-amz-date"));
        assert!(reserved.contains(&"x-amz-content-sha256"));
        assert!(reserved.contains(&"x-amz-security-token"));
    }

    #[test]
    fn reserved_sigv4_headers_includes_bedrock_specific() {
        // Bedrock's event-stream content-type negotiation is a
        // wire-shape gate. Pinning so the override-guard list stays
        // honest about Bedrock specifics, not just generic SigV4.
        let reserved = reserved_sigv4_headers();
        assert!(reserved.contains(&"x-amzn-bedrock-accept"));
    }
}
