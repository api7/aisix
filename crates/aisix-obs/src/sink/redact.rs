//! Redaction of a sink's own configured secrets from its delivery-error text.

/// Shortest trailing fragment of a secret [`ErrorRedactor::redact`] treats
/// as that secret cut off by a cap.
const MIN_CUT_PREFIX: usize = 4;

/// The configured values one sink must never surface in its error text, and
/// what each is rendered as instead.
///
/// Delivery errors reach the warn log, `SinkStats::last_error` and, through
/// it, the managed-mode heartbeat's `exporter_health[].last_error`. Their
/// text is not the sink's to control: a receiver can echo a credential back
/// in its response body. So each sink declares the secrets it was
/// configured with and the pipeline scrubs every error through this once,
/// before any of those surfaces reads it.
///
/// Matching is exact, on the configured value: text the receiver wrote is
/// otherwise kept as-is.
#[derive(Debug, Clone, Default)]
pub struct ErrorRedactor {
    /// `(needle, replacement)`, longest needle first so a secret that
    /// contains another is replaced whole.
    rules: Vec<(String, &'static str)>,
}

impl ErrorRedactor {
    /// Redact a configured secret value (a key, a token, a header value)
    /// wherever it appears.
    pub fn secret(mut self, value: &str) -> Self {
        self.push(value.to_string(), "***");
        self
    }

    fn push(&mut self, needle: String, replacement: &'static str) {
        if needle.is_empty() || self.rules.iter().any(|(n, _)| *n == needle) {
            return;
        }
        let at = self
            .rules
            .iter()
            .position(|(n, _)| n.len() < needle.len())
            .unwrap_or(self.rules.len());
        self.rules.insert(at, (needle, replacement));
    }

    /// `text` with every configured secret replaced.
    ///
    /// The sinks cap their own detail before it gets here, always by
    /// cutting the end off, so a secret the cut went through survives only
    /// as a prefix at the very end of `text`; that tail is replaced too once
    /// it is long enough to mean anything, so an error that merely ends in
    /// a secret's first character keeps it.
    pub fn redact(&self, text: &str) -> String {
        let mut out = text.to_string();
        for (needle, replacement) in &self.rules {
            if out.contains(needle.as_str()) {
                out = out.replace(needle.as_str(), replacement);
            }
        }
        let cut = self
            .rules
            .iter()
            .filter_map(|(needle, replacement)| {
                let longest = needle
                    .char_indices()
                    .map(|(i, _)| &needle[..i])
                    .rfind(|p| p.len() >= MIN_CUT_PREFIX && out.ends_with(p))?;
                Some((longest.len(), *replacement))
            })
            .max_by_key(|(len, _)| *len);
        if let Some((len, replacement)) = cut {
            out.truncate(out.len() - len);
            out.push_str(replacement);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::ErrorRedactor;

    #[test]
    fn a_secret_cut_off_at_the_end_of_the_text_is_replaced() {
        let r = ErrorRedactor::default().secret("otlp-header-token");
        assert_eq!(
            r.redact("HTTP 401: rejected otlp-hea"),
            "HTTP 401: rejected ***"
        );
        assert_eq!(r.redact("HTTP 401: rejected"), "HTTP 401: rejected");
        assert_eq!(r.redact("HTTP 401: invalid o"), "HTTP 401: invalid o");
    }

    #[test]
    fn a_secret_containing_another_is_replaced_whole() {
        let r = ErrorRedactor::default()
            .secret("tok")
            .secret("Bearer tok-long");
        assert_eq!(r.redact("echo: Bearer tok-long / tok"), "echo: *** / ***");
    }
}
