//! Entry-level URL rewriting (`proxy.url_rewrites`).
//!
//! Applied to every proxy-listener request **before** route matching (the
//! admin and metrics listeners never see this layer): the first rule whose
//! `match` regex matches the request path rewrites it — once, no cascading —
//! and the request then flows through the normal endpoint (auth, ACL, quota,
//! metrics labelling) as if the client had sent the rewritten path. A miss
//! leaves the request untouched.
//!
//! Replacement substitutes the **matched portion** of the path, with
//! `$1`/`${name}` expanding capture groups; the query string is preserved as
//! sent. Rules were syntax-checked at config load (`Config::validate`), so
//! compiling them here cannot fail on operator input.

use std::borrow::Cow;
use std::sync::Arc;

use axum::extract::{Request, State};
use axum::http;
use axum::middleware::Next;
use axum::response::Response;
use regex::Regex;

use aisix_core::UrlRewriteRule;

use crate::state::ProxyState;

/// One boot-compiled rewrite rule.
pub struct CompiledRewrite {
    name: Option<String>,
    pattern: Regex,
    replacement: String,
}

impl CompiledRewrite {
    /// The rewritten path, or `None` when the rule does not match.
    fn apply(&self, path: &str) -> Option<String> {
        match self.pattern.replace(path, self.replacement.as_str()) {
            // `replace` hands the haystack back untouched on a miss.
            Cow::Borrowed(_) => None,
            Cow::Owned(rewritten) => Some(rewritten),
        }
    }

    fn label(&self) -> &str {
        self.name.as_deref().unwrap_or("<unnamed>")
    }
}

/// Compile the configured rules in declaration order.
///
/// Invariant: every pattern was syntax-checked by `Config::validate` at
/// load, so a panic here means a caller constructed a `ProxyConfig` with an
/// unvalidated pattern.
pub fn compile(rules: &[UrlRewriteRule]) -> Arc<[CompiledRewrite]> {
    rules
        .iter()
        .map(|rule| CompiledRewrite {
            name: rule.name.clone(),
            pattern: Regex::new(&rule.pattern)
                .expect("proxy.url_rewrites pattern is validated at config load"),
            replacement: rule.replacement.clone(),
        })
        .collect()
}

/// Middleware: apply the first matching rewrite rule to the request path.
pub async fn rewrite_request_uri(
    State(state): State<ProxyState>,
    mut request: Request,
    next: Next,
) -> Response {
    if state.url_rewrites.is_empty() {
        return next.run(request).await;
    }
    let fired = state.url_rewrites.iter().find_map(|rule| {
        let path = request.uri().path();
        rule.apply(path).map(|to| (rule, path.to_owned(), to))
    });
    if let Some((rule, from, to)) = fired {
        match with_path(request.uri(), &to) {
            Ok(uri) => {
                tracing::debug!(rule = rule.label(), %from, %to, "url rewrite applied");
                *request.uri_mut() = uri;
            }
            Err(error) => {
                // A template can assemble an invalid path out of matched
                // input (e.g. an empty string). Serving the original path is
                // the conservative outcome: the request 404s the same way it
                // would have without the layer, and the warn names the rule.
                tracing::warn!(
                    rule = rule.label(),
                    %from,
                    rewritten = %to,
                    %error,
                    "url rewrite produced an invalid path; leaving the request unrewritten"
                );
            }
        }
    }
    next.run(request).await
}

/// `uri` with its path replaced by `new_path`, query preserved.
fn with_path(uri: &http::Uri, new_path: &str) -> Result<http::Uri, http::Error> {
    let path_and_query = match uri.query() {
        Some(query) => format!("{new_path}?{query}"),
        None => new_path.to_owned(),
    };
    let mut parts = uri.clone().into_parts();
    parts.path_and_query = Some(path_and_query.parse()?);
    Ok(http::Uri::from_parts(parts)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rule(pattern: &str, replacement: &str) -> UrlRewriteRule {
        UrlRewriteRule {
            name: None,
            pattern: pattern.to_string(),
            replacement: replacement.to_string(),
        }
    }

    #[test]
    fn apply_substitutes_capture_groups() {
        let compiled = compile(&[rule("^/mcp-servers/([^/]+)/mcp$", "/mcp/$1")]);
        assert_eq!(
            compiled[0].apply("/mcp-servers/github/mcp").as_deref(),
            Some("/mcp/github")
        );
        assert_eq!(compiled[0].apply("/mcp-servers/github/sse"), None);
        assert_eq!(compiled[0].apply("/v1/chat/completions"), None);
    }

    #[test]
    fn apply_replaces_only_the_matched_portion() {
        // Unanchored pattern: the unmatched prefix survives, mirroring the
        // replace-matched-portion semantics of mainstream gateways.
        let compiled = compile(&[rule("/legacy$", "/current")]);
        assert_eq!(
            compiled[0].apply("/api/legacy").as_deref(),
            Some("/api/current")
        );
    }

    #[test]
    fn apply_supports_named_groups_and_braced_references() {
        let compiled = compile(&[rule(
            "^/gw/(?P<server>[^/]+)/v(\\d+)$",
            "/mcp/${server}-v${2}",
        )]);
        assert_eq!(
            compiled[0].apply("/gw/github/v2").as_deref(),
            Some("/mcp/github-v2")
        );
    }

    #[test]
    fn with_path_preserves_the_query_string() {
        let uri: http::Uri = "http://gw.example/mcp-servers/github/mcp?probe=1"
            .parse()
            .unwrap();
        let rewritten = with_path(&uri, "/mcp/github").unwrap();
        assert_eq!(rewritten.path(), "/mcp/github");
        assert_eq!(rewritten.query(), Some("probe=1"));
        assert_eq!(rewritten.host(), Some("gw.example"));
    }

    #[test]
    fn with_path_rejects_an_invalid_path() {
        let uri: http::Uri = "/x".parse().unwrap();
        // A template can assemble characters that are invalid in a request
        // path; the middleware then keeps the original URI (warn + no-op).
        assert!(with_path(&uri, "/a b").is_err());
    }

    fn router_with_rules(rules: Vec<UrlRewriteRule>) -> axum::Router {
        use aisix_core::snapshot::SnapshotHandle;
        let cfg = aisix_core::ProxyConfig {
            addr: "127.0.0.1:0".into(),
            request_body_limit_bytes: 0,
            tls: None,
            real_ip: Default::default(),
            url_rewrites: rules,
        };
        let state = ProxyState::new(
            SnapshotHandle::new(aisix_core::AisixSnapshot::new()),
            Arc::new(aisix_gateway::Hub::new()),
            &cfg,
        )
        .without_cache();
        crate::build_router(state)
    }

    async fn get(router: axum::Router, path: &str) -> axum::http::StatusCode {
        use tower::ServiceExt;
        let request = axum::http::Request::get(path)
            .body(axum::body::Body::empty())
            .unwrap();
        router.oneshot(request).await.unwrap().status()
    }

    #[tokio::test]
    async fn router_serves_a_rewritten_legacy_path_and_first_rule_wins() {
        // Two rules match the legacy path; the first rewrites onto a real
        // endpoint, the second onto a 404. Declaration order must win.
        let router = router_with_rules(vec![
            rule("^/legacy/health$", "/livez"),
            rule("^/legacy/health$", "/nonexistent"),
        ]);
        assert_eq!(get(router.clone(), "/legacy/health").await, 200);
        // A miss flows through unrewritten.
        assert_eq!(get(router.clone(), "/legacy/other").await, 404);
        // The canonical path keeps working alongside the legacy one.
        assert_eq!(get(router, "/livez").await, 200);
    }

    #[tokio::test]
    async fn router_without_rules_is_untouched() {
        let router = router_with_rules(Vec::new());
        assert_eq!(get(router.clone(), "/livez").await, 200);
        assert_eq!(get(router, "/legacy/health").await, 404);
    }
}
