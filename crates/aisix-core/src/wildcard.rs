//! Single-`*` wildcard matching, shared by API-key model allowlists
//! ([`crate::models::ApiKey::can_access`]) and the proxy's wildcard model
//! resolution (`provider/*` aliases).
//!
//! Only one `*` per pattern is supported — enough for the common
//! `provider/*` / `prefix-*-suffix` shapes and avoids pulling a regex engine
//! into the hot path. The literals on either side of the `*` anchor the
//! candidate at both ends, so a pattern matches the *whole* name.

/// Match `candidate` against a `*`-glob `pattern` that contains **exactly one**
/// `*`, returning the substring bound to the `*` on success.
///
/// `openai/*` matches `openai/gpt-4o` (capture `gpt-4o`) but not
/// `azure/openai/x` (the prefix must anchor the start). A pattern with no `*`,
/// or with more than one `*`, is unsupported and returns `None` — callers treat
/// those as plain literals.
pub fn wildcard_capture(pattern: &str, candidate: &str) -> Option<String> {
    let star = pattern.find('*')?;
    if pattern[star + 1..].contains('*') {
        return None; // only a single '*' is supported
    }
    let prefix = &pattern[..star];
    let suffix = &pattern[star + 1..];
    if candidate.len() < prefix.len() + suffix.len() {
        return None; // prefix and suffix would overlap
    }
    if candidate.starts_with(prefix) && candidate.ends_with(suffix) {
        Some(candidate[prefix.len()..candidate.len() - suffix.len()].to_string())
    } else {
        None
    }
}

/// Whether `pattern` matches `candidate`. A pattern with a single `*` is
/// glob-matched; any other pattern matches only an exactly-equal candidate.
pub fn wildcard_matches(pattern: &str, candidate: &str) -> bool {
    if pattern.contains('*') {
        wildcard_capture(pattern, candidate).is_some()
    } else {
        pattern == candidate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trailing_star_captures_the_rest() {
        assert_eq!(
            wildcard_capture("openai/*", "openai/gpt-4o").as_deref(),
            Some("gpt-4o")
        );
        assert_eq!(
            wildcard_capture("openai/*", "openai/gpt-4o-mini").as_deref(),
            Some("gpt-4o-mini")
        );
    }

    #[test]
    fn prefix_must_anchor_the_start() {
        assert_eq!(wildcard_capture("openai/*", "azure/openai/x"), None);
        assert_eq!(wildcard_capture("openai/*", "opena"), None);
    }

    #[test]
    fn interior_star_binds_the_middle() {
        assert_eq!(
            wildcard_capture("gpt-*-preview", "gpt-4o-preview").as_deref(),
            Some("4o")
        );
        assert_eq!(wildcard_capture("gpt-*-preview", "gpt-4o-final"), None);
    }

    #[test]
    fn star_matches_empty_capture() {
        assert_eq!(wildcard_capture("openai/*", "openai/").as_deref(), Some(""));
    }

    #[test]
    fn multiple_stars_unsupported() {
        assert_eq!(wildcard_capture("a/*/*", "a/b/c"), None);
    }

    #[test]
    fn matches_handles_literals_and_globs() {
        assert!(wildcard_matches("*", "anything"));
        assert!(wildcard_matches("openai/*", "openai/gpt-4o"));
        assert!(wildcard_matches("gpt-4o", "gpt-4o"));
        assert!(!wildcard_matches("gpt-4o", "gpt-4o-mini"));
        assert!(!wildcard_matches("openai/*", "anthropic/claude"));
    }
}
