//! Host patterns shared by entry-level rewrites and passthrough routes.

/// Exact host, or a single-label wildcard with at least two literal labels.
pub const HOST_PATTERN: &str = r"^(\*\.)?([A-Za-z0-9-]+\.)+[A-Za-z0-9-]+$|^[A-Za-z0-9-]+$";

/// Match a normalized host (lowercase, no port) against configured patterns.
pub fn matches(hosts: &[String], host: &str) -> bool {
    hosts.iter().any(|pattern| {
        let p = pattern.to_ascii_lowercase();
        if let Some(suffix) = p.strip_prefix("*.") {
            match host.strip_suffix(suffix) {
                Some(head) => {
                    head.ends_with('.')
                        && !head[..head.len() - 1].is_empty()
                        && !head[..head.len() - 1].contains('.')
                }
                None => false,
            }
        } else {
            p == host
        }
    })
}
