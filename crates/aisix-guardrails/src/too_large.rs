//! Recognising "the provider refused this payload for being too big".
//!
//! Every remote kind already classifies its failures, but none of them
//! had a bucket for this one, so an over-limit refusal was filed as
//! whatever its status code happened to look like — a throttle, or a
//! misconfiguration. Both are actively misleading: an operator who reads
//! `throttled` waits and retries, and the retry can never succeed
//! because the cause is the size of the request, not its rate
//! (AISIX-Cloud#1386).
//!
//! The bound itself is deliberately NOT hardcoded per kind. Of the four
//! kinds with no local limit, not one publishes a figure we could hold:
//!
//! - **Bedrock** caps `ApplyGuardrail` at 25–1 000 text units of 1 000
//!   characters each, varying by region, by policy type, by tier, and
//!   adjustable per account — so between 25 000 and 1 000 000 characters,
//!   knowable only to the account that owns the quota.
//!   <https://docs.aws.amazon.com/general/latest/gr/bedrock.html>
//! - **OpenAI** documents no input limit for `/moderations` at all; the
//!   real constraint is the tokens-per-minute budget, which a single
//!   large request exhausts — which is why the refusal arrives as a 429
//!   whose message is about rate.
//!   <https://developers.openai.com/api/docs/guides/moderation>
//! - **Presidio** has no limit of its own — the ceiling is whichever
//!   spaCy pipeline the operator deployed (`nlp.max_length`, 1 000 000
//!   characters by default), i.e. their configuration, not ours.
//!   <https://github.com/microsoft/presidio/discussions/916>
//! - **Lakera** publishes nothing: neither a size limit nor an error
//!   response other than `200`.
//!   <https://docs.lakera.ai/api-reference/lakera-api/guard/screen-content>
//!
//! So detection is reactive by necessity, and the evidence for it is
//! uneven. HTTP 413 is the standard answer and costs nothing to honour
//! anywhere. Beyond that only Bedrock and Presidio have a documented
//! message shape to match; for Lakera and OpenAI we match nothing
//! extra rather than invent vendor strings we have never seen.

/// Wording a provider uses when it refuses a payload for its size.
///
/// Sourced, not guessed: the first six are AWS's, the last two are
/// spaCy's `E088` (which surfaces through Presidio's analyzer when the
/// text exceeds the deployed pipeline's `nlp.max_length`). Matching is
/// case-insensitive and substring-based because these arrive inside
/// longer sentences that also carry the offending numbers.
const TOO_LARGE_MARKERS: &[&str] = &[
    "text unit",
    "maximum input size",
    "content size",
    "too long",
    "too large",
    "exceeds the maximum",
    "exceeds maximum of",
    "e088",
];

/// Whether a provider's error text says the payload was too big.
///
/// Status is handled at the call sites, which match `413` directly:
/// that is the only status meaning this unambiguously. `400` and `429`
/// are deliberately NOT treated as size failures by status — both have
/// overwhelmingly more common causes (a malformed request, a real
/// throttle), so they reach this function and are judged on their
/// message instead.
///
/// Callers pass the response body or the SDK error's own rendering; both
/// are bounded upstream before reaching here.
pub(crate) fn body_says_too_large(body: &str) -> bool {
    let lowered = body.to_ascii_lowercase();
    TOO_LARGE_MARKERS.iter().any(|m| lowered.contains(m))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// AWS's own wording, which is what makes the Bedrock reactive split
    /// fire. Taken from the quota documentation's vocabulary.
    #[test]
    fn recognises_aws_size_wording() {
        for body in [
            "ValidationException: Input is too long for requested model",
            "maximum input size in text units exceeded",
            "The content size exceeds the maximum allowed",
            "Guardrail input too large",
            "input exceeds the maximum of 25 text units",
        ] {
            assert!(body_says_too_large(body), "missed: {body}");
        }
    }

    /// spaCy's error, which is what a Presidio analyzer raises when the
    /// text is longer than the deployed pipeline allows.
    #[test]
    fn recognises_spacy_max_length_error() {
        let body = "ValueError: [E088] Text of length 2000000 exceeds maximum of 1000000. \
                    The parser and NER models require roughly 1GB of temporary memory per \
                    100,000 characters in the input.";
        assert!(body_says_too_large(body));
    }

    #[test]
    fn matching_is_case_insensitive() {
        assert!(body_says_too_large("CONTENT SIZE EXCEEDS LIMIT"));
        assert!(body_says_too_large("Text Unit quota exceeded"));
    }

    /// The failure texts that must keep their existing buckets — a
    /// mis-hit here would relabel a real outage or misconfiguration as a
    /// size problem and send the operator after the wrong fix.
    #[test]
    fn ordinary_failures_are_not_size_failures() {
        for body in [
            "AccessDeniedException: not authorized to perform bedrock:ApplyGuardrail",
            "ThrottlingException: Rate exceeded",
            "Rate limit reached for text-moderation-007 on tokens per min (TPM)",
            "invalid_api_key: Incorrect API key provided",
            "503 Service Unavailable",
            "connection reset by peer",
        ] {
            assert!(!body_says_too_large(body), "false positive: {body}");
        }
    }
}
