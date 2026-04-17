//! Parse etcd keys of the shape `{prefix}/{kind}/{id}`.
//!
//! Every aisix entity is stored at this canonical path. The watch supervisor
//! demultiplexes incoming events by the `kind` segment (`models`, `apikeys`,
//! `teams`, …) so each typed table can be updated independently.

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceKey<'a> {
    pub kind: &'a str,
    pub id: &'a str,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum KeyError {
    #[error("etcd key {key:?} does not start with configured prefix {prefix:?}")]
    PrefixMismatch { key: String, prefix: String },
    #[error("etcd key {0:?} is missing the `{{kind}}/{{id}}` suffix")]
    MissingSuffix(String),
    #[error("etcd key {0:?} has an empty kind or id segment")]
    EmptySegment(String),
}

/// Split an etcd key into (kind, id) given the configured aisix prefix.
///
/// Example: with prefix `/aisix`, a key `/aisix/models/abc-123` parses to
/// `ResourceKey { kind: "models", id: "abc-123" }`.
pub fn parse<'a>(prefix: &str, key: &'a str) -> Result<ResourceKey<'a>, KeyError> {
    // Accept both `/aisix` and `/aisix/` prefixes transparently.
    let trimmed_prefix = prefix.trim_end_matches('/');
    let rest = key
        .strip_prefix(trimmed_prefix)
        .ok_or_else(|| KeyError::PrefixMismatch {
            key: key.to_string(),
            prefix: prefix.to_string(),
        })?;
    let rest = rest.trim_start_matches('/');

    let (kind, id) = rest
        .split_once('/')
        .ok_or_else(|| KeyError::MissingSuffix(key.to_string()))?;

    if kind.is_empty() || id.is_empty() {
        return Err(KeyError::EmptySegment(key.to_string()));
    }

    Ok(ResourceKey { kind, id })
}

impl fmt::Display for ResourceKey<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.kind, self.id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn happy_path_parses_kind_and_id() {
        let k = parse("/aisix", "/aisix/models/abc-123").unwrap();
        assert_eq!(k.kind, "models");
        assert_eq!(k.id, "abc-123");
    }

    #[test]
    fn trailing_slash_in_prefix_is_tolerated() {
        let k = parse("/aisix/", "/aisix/apikeys/uuid-1").unwrap();
        assert_eq!(k.kind, "apikeys");
        assert_eq!(k.id, "uuid-1");
    }

    #[test]
    fn prefix_mismatch_is_detected() {
        let err = parse("/aisix", "/other/models/a").unwrap_err();
        assert!(matches!(err, KeyError::PrefixMismatch { .. }));
    }

    #[test]
    fn missing_suffix_is_rejected() {
        // Prefix-only key, no kind/id.
        let err = parse("/aisix", "/aisix/models").unwrap_err();
        assert!(matches!(err, KeyError::MissingSuffix(_)));
    }

    #[test]
    fn empty_segments_are_rejected() {
        let err = parse("/aisix", "/aisix/models/").unwrap_err();
        assert!(matches!(err, KeyError::EmptySegment(_)));
    }

    #[test]
    fn display_is_kind_slash_id() {
        let k = ResourceKey {
            kind: "models",
            id: "abc",
        };
        assert_eq!(k.to_string(), "models/abc");
    }
}
