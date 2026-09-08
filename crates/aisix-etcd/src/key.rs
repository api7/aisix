//! Parse etcd keys of the shape `{prefix}/{kind}/{id}`.
//!
//! Every aisix entity is stored at this canonical path. The watch supervisor
//! demultiplexes incoming events by the `kind` segment (`models`, `api_keys`,
//! `provider_keys`, `guardrails`, …) so each typed table can be updated
//! independently.
//!
//! The supervisor watches more than one prefix — its environment's, plus
//! the shared `<base>/global/` catalog — over one snapshot, so a key is
//! resolved against a [`PrefixSet`] rather than a single string. Which
//! prefix a key came from is part of its identity: the same `pricing`
//! kind means two different tables depending on the prefix it was
//! written under.

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
    // Enforce a delimiter boundary after the prefix: the next character must
    // be `/`. Without this, `strip_prefix` byte-matching would treat an
    // adjacent key such as `/aisixmodels/x` as if it lived under `/aisix/`,
    // letting a writer outside the configured namespace inject models or API
    // keys. An exact-prefix match (`rest == ""`) falls through to the
    // MissingSuffix check below.
    if !rest.is_empty() && !rest.starts_with('/') {
        return Err(KeyError::PrefixMismatch {
            key: key.to_string(),
            prefix: prefix.to_string(),
        });
    }
    let rest = rest.strip_prefix('/').unwrap_or(rest);

    let (kind, id) = rest
        .split_once('/')
        .ok_or_else(|| KeyError::MissingSuffix(key.to_string()))?;

    if kind.is_empty() || id.is_empty() {
        return Err(KeyError::EmptySegment(key.to_string()));
    }

    Ok(ResourceKey { kind, id })
}

/// Which of the supervisor's prefixes a key was written under.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefixScope {
    /// The gateway's own environment: `<base>/<env_id>/`. Carries every
    /// resource kind.
    Environment,
    /// The shared catalog: `<base>/global/`. Carries `pricing` and
    /// nothing else — see [`crate::loader::build_snapshot`].
    Global,
}

impl PrefixScope {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Environment => "environment",
            Self::Global => "global",
        }
    }
}

/// One prefix the supervisor watches, with what it is allowed to carry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WatchedPrefix {
    pub prefix: String,
    pub scope: PrefixScope,
}

impl WatchedPrefix {
    pub fn environment(prefix: impl Into<String>) -> Self {
        Self {
            prefix: prefix.into(),
            scope: PrefixScope::Environment,
        }
    }

    pub fn global(prefix: impl Into<String>) -> Self {
        Self {
            prefix: prefix.into(),
            scope: PrefixScope::Global,
        }
    }
}

/// The prefixes one supervisor resolves keys against.
///
/// Ordered longest-first, so a key is attributed to the most specific
/// prefix that contains it. That matters when the environment prefix is
/// the bare base — the pre-`env_id` shape a self-managed deployment still
/// uses — because `<base>/global/` then nests inside it, and a global
/// document resolved against the outer prefix would parse as kind
/// `global` and be rejected.
#[derive(Debug, Clone)]
pub struct PrefixSet {
    prefixes: Vec<WatchedPrefix>,
}

impl PrefixSet {
    pub fn new(mut prefixes: Vec<WatchedPrefix>) -> Self {
        prefixes.sort_by_key(|p| std::cmp::Reverse(p.prefix.trim_end_matches('/').len()));
        Self { prefixes }
    }

    /// A set holding one environment prefix — the shape every caller that
    /// does not read the shared catalog uses.
    pub fn single(prefix: impl Into<String>) -> Self {
        Self::new(vec![WatchedPrefix::environment(prefix)])
    }

    pub fn iter(&self) -> impl Iterator<Item = &WatchedPrefix> {
        self.prefixes.iter()
    }

    pub fn len(&self) -> usize {
        self.prefixes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.prefixes.is_empty()
    }

    /// Split `key` into scope, kind and id. Reports the mismatch against
    /// the first (longest) prefix, which is the one an operator is most
    /// likely to have meant.
    pub fn resolve<'a>(&self, key: &'a str) -> Result<ScopedKey<'a>, KeyError> {
        let mut first_err = None;
        for watched in &self.prefixes {
            match parse(&watched.prefix, key) {
                Ok(parsed) => {
                    return Ok(ScopedKey {
                        scope: watched.scope,
                        kind: parsed.kind,
                        id: parsed.id,
                    })
                }
                // A key under this prefix with a malformed suffix is a bad
                // key, not a reason to try the next prefix: no other
                // prefix contains it either.
                Err(err @ (KeyError::MissingSuffix(_) | KeyError::EmptySegment(_))) => {
                    return Err(err)
                }
                Err(err) => first_err.get_or_insert(err),
            };
        }
        Err(first_err.unwrap_or_else(|| KeyError::PrefixMismatch {
            key: key.to_string(),
            prefix: String::new(),
        }))
    }
}

/// A parsed key plus the prefix scope it came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScopedKey<'a> {
    pub scope: PrefixScope,
    /// The `kind` segment exactly as written in the key. This is what
    /// rejection and compatibility reports quote, so the control plane
    /// sees the name it wrote.
    pub kind: &'a str,
    pub id: &'a str,
}

impl<'a> ScopedKey<'a> {
    /// The snapshot table this row belongs in.
    ///
    /// Differs from [`ScopedKey::kind`] for exactly one pair: a `pricing`
    /// document under the global prefix belongs in `global_pricing`, so
    /// an environment document carrying the same `key` can win over it.
    /// Every snapshot-shaped operation — insert, remove, presence probe,
    /// per-kind counts — keys on this, never on the raw kind.
    pub fn table_kind(&self) -> &'a str {
        match (self.scope, self.kind) {
            (PrefixScope::Global, "pricing") => "global_pricing",
            _ => self.kind,
        }
    }
}

impl fmt::Display for ScopedKey<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.kind, self.id)
    }
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
        let k = parse("/aisix/", "/aisix/api_keys/uuid-1").unwrap();
        assert_eq!(k.kind, "api_keys");
        assert_eq!(k.id, "uuid-1");
    }

    #[test]
    fn prefix_mismatch_is_detected() {
        let err = parse("/aisix", "/other/models/a").unwrap_err();
        assert!(matches!(err, KeyError::PrefixMismatch { .. }));
    }

    #[test]
    fn adjacent_prefix_without_delimiter_is_rejected() {
        // `/aisixmodels/...` byte-starts with `/aisix` but is NOT a child of
        // the `/aisix/` namespace — it must not be parsed as config.
        for key in [
            "/aisixmodels/models/m-1",
            "/aisixapikeys/api_keys/k-1",
            "/aisix-extra/models/m-1",
            "/aisixfoo/bar/baz",
        ] {
            let err = parse("/aisix", key).unwrap_err();
            assert!(
                matches!(err, KeyError::PrefixMismatch { .. }),
                "expected PrefixMismatch for {key:?}, got {err:?}",
            );
        }
    }

    #[test]
    fn child_of_namespace_is_accepted() {
        let k = parse("/aisix", "/aisix/models/m-1").unwrap();
        assert_eq!(k.kind, "models");
        assert_eq!(k.id, "m-1");
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

    fn set(env: &str, global: &str) -> PrefixSet {
        PrefixSet::new(vec![
            WatchedPrefix::environment(env),
            WatchedPrefix::global(global),
        ])
    }

    #[test]
    fn a_key_resolves_to_the_prefix_that_holds_it() {
        let s = set("/aisix/env-1/", "/aisix/global/");
        let env = s.resolve("/aisix/env-1/models/m-1").unwrap();
        assert_eq!(env.scope, PrefixScope::Environment);
        assert_eq!((env.kind, env.id), ("models", "m-1"));

        let global = s.resolve("/aisix/global/pricing/p-1").unwrap();
        assert_eq!(global.scope, PrefixScope::Global);
        assert_eq!((global.kind, global.id), ("pricing", "p-1"));
    }

    #[test]
    fn the_global_prefix_wins_when_it_nests_inside_a_bare_environment_prefix() {
        // The pre-`env_id` shape a self-managed deployment still uses:
        // the environment prefix is the bare base, so `<base>/global/`
        // sits INSIDE it. Resolved against the outer prefix the key would
        // parse as kind `global` and be rejected, silently costing every
        // model its catalog price.
        let s = set("/aisix", "/aisix/global/");
        let k = s.resolve("/aisix/global/pricing/p-1").unwrap();
        assert_eq!(k.scope, PrefixScope::Global);
        assert_eq!((k.kind, k.id), ("pricing", "p-1"));

        // A sibling of `global` under the same base is still the
        // environment's.
        let m = s.resolve("/aisix/models/m-1").unwrap();
        assert_eq!(m.scope, PrefixScope::Environment);
        assert_eq!(m.kind, "models");
    }

    #[test]
    fn a_key_under_no_watched_prefix_is_a_mismatch() {
        let s = set("/aisix/env-1/", "/aisix/global/");
        let err = s.resolve("/aisix/env-2/models/m-1").unwrap_err();
        assert!(matches!(err, KeyError::PrefixMismatch { .. }), "{err:?}");
    }

    #[test]
    fn a_malformed_suffix_under_a_watched_prefix_is_not_retried_elsewhere() {
        let s = set("/aisix/env-1/", "/aisix/global/");
        assert!(matches!(
            s.resolve("/aisix/env-1/models").unwrap_err(),
            KeyError::MissingSuffix(_)
        ));
        assert!(matches!(
            s.resolve("/aisix/global/pricing/").unwrap_err(),
            KeyError::EmptySegment(_)
        ));
    }

    #[test]
    fn only_global_pricing_takes_a_different_table_than_its_kind() {
        let s = set("/aisix/env-1/", "/aisix/global/");
        assert_eq!(
            s.resolve("/aisix/global/pricing/p-1").unwrap().table_kind(),
            "global_pricing"
        );
        // The same kind under the environment keeps its own table, which
        // is what lets an environment document override a catalog one.
        assert_eq!(
            s.resolve("/aisix/env-1/pricing/p-1").unwrap().table_kind(),
            "pricing"
        );
        assert_eq!(
            s.resolve("/aisix/env-1/models/m-1").unwrap().table_kind(),
            "models"
        );
    }

    #[test]
    fn a_single_prefix_set_is_environment_scoped() {
        let s = PrefixSet::single("/aisix");
        assert_eq!(s.len(), 1);
        assert_eq!(
            s.resolve("/aisix/models/m-1").unwrap().scope,
            PrefixScope::Environment
        );
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
