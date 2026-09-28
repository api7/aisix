//! Redaction of a sink's own configured secrets from its delivery-error text.

use std::borrow::Cow;

use aisix_core::redact_url_userinfo;

/// The configured values one sink must never surface in its error text, and
/// what each is rendered as instead.
///
/// Delivery errors reach the warn log, `SinkStats::last_error` and, through
/// it, the managed-mode heartbeat's `exporter_health[].last_error`. Their
/// text is not the sink's to control: a transport error names the configured
/// URL, object_store renders the request URI with its userinfo, and a
/// receiver can echo a credential back in its response body. So each sink
/// declares what it was configured with and the pipeline scrubs every error
/// through this once, before any of those surfaces reads it.
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
    /// Redact the userinfo of a configured URL, both as written and as the
    /// URL parser re-serializes it (which percent-encodes some characters,
    /// and is how an HTTP client renders the request URI). The password is
    /// also redacted on its own.
    pub fn url(mut self, url: &str) -> Self {
        // The shared helper decides what counts as userinfo; its output
        // locates the span it replaced.
        let Cow::Owned(redacted) = redact_url_userinfo(url) else {
            return self;
        };
        let Some(at) = redacted.find("***@") else {
            return self;
        };
        let tail = redacted.len() - at - "***".len();
        let userinfo = &url[at..url.len() - tail];
        self.push(format!("{userinfo}@"), "***@");
        if let Some((_, password)) = userinfo.split_once(':') {
            self.push(password.to_string(), "***");
        }
        if let Ok(parsed) = reqwest::Url::parse(url) {
            let user = parsed.username();
            match parsed.password() {
                Some(password) => {
                    self.push(format!("{user}:{password}@"), "***@");
                    self.push(password.to_string(), "***");
                }
                None if !user.is_empty() => self.push(format!("{user}@"), "***@"),
                None => {}
            }
        }
        self
    }

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
    /// as a prefix at the very end of `text`; that tail is replaced too.
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
                    .rfind(|p| !p.is_empty() && out.ends_with(p))?;
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
    fn redacts_userinfo_as_written_and_as_the_url_parser_renders_it() {
        let r = ErrorRedactor::default().url("https://svc:p@ss w0rd@collector.example/v1/traces");
        for rendered in [
            "POST https://svc:p@ss w0rd@collector.example/v1/traces: timed out",
            "Error performing PUT https://svc:p%40ss%20w0rd@collector.example/v1/traces/x",
        ] {
            let out = r.redact(rendered);
            assert!(!out.contains("p@ss") && !out.contains("p%40ss"), "{out}");
            assert!(out.contains("***@collector.example/v1/traces"), "{out}");
        }
    }

    #[test]
    fn leaves_a_url_without_userinfo_and_unrelated_text_alone() {
        let r = ErrorRedactor::default()
            .url("https://collector.example/v1/traces?tenant=a@b")
            .secret("");
        let text = "POST https://collector.example/v1/traces?tenant=a@b: HTTP 503";
        assert_eq!(r.redact(text), text);
    }

    #[test]
    fn a_secret_cut_off_at_the_end_of_the_text_is_replaced() {
        let r = ErrorRedactor::default().secret("otlp-header-token");
        assert_eq!(
            r.redact("HTTP 401: rejected otlp-hea"),
            "HTTP 401: rejected ***"
        );
        assert_eq!(r.redact("HTTP 401: rejected"), "HTTP 401: rejected");
    }

    #[test]
    fn a_secret_containing_another_is_replaced_whole() {
        let r = ErrorRedactor::default()
            .secret("tok")
            .secret("Bearer tok-long");
        assert_eq!(r.redact("echo: Bearer tok-long / tok"), "echo: *** / ***");
    }
}
