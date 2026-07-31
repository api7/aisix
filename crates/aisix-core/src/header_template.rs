//! `${...}` variable substitution for `ProviderKey.request.default_headers`
//! values (AISIX-Cloud#1112).
//!
//! An operator writes `"x-tenant-id": "${request.api_key.team_id}"` on the
//! ProviderKey and the data plane renders it per request, just before the
//! upstream call, so internal model services can attribute traffic to the
//! calling tenant/team/key without the customer minting a provider
//! credential per tenant.
//!
//! Three rules make this safe to expose to operator-supplied config:
//!
//! 1. **Closed vocabulary.** Only the names in [`HEADER_TEMPLATE_VARS`]
//!    resolve. Anything else makes the whole template unresolvable — cp-api
//!    rejects unknown names at write time and the renderer refuses them
//!    again at runtime, so a typo can never fall through as a literal
//!    `${...}` on the wire.
//! 2. **No secret is reachable.** The vocabulary carries identifiers and
//!    display names only. The caller's plaintext bearer, the ProviderKey
//!    secret and every credential field are simply not in [`HeaderVars`],
//!    so no template can name them.
//! 3. **Empty means skip, never blank.** A variable that has no value for
//!    this request (a key with no team, say) renders the whole header
//!    unresolvable and the header is dropped. Emitting `x-tenant-id: ""`
//!    would read to the upstream as "tenant is the empty string", which is
//!    a different and misleading claim.

use std::fmt::Write as _;

/// The closed set of variable names a `default_headers` value may
/// reference. Kept in the same order the docs list them.
///
/// cp-api mirrors this list in
/// `internal/cpapi/resources/provider_key_overrides.go` so a bad template
/// is a 400 at write time rather than a silently-dropped header at
/// dispatch time. **The two lists must stay in sync**; this one is
/// canonical.
pub const HEADER_TEMPLATE_VARS: &[&str] = &[
    "request.id",
    "request.api_key.id",
    "request.api_key.name",
    "request.api_key.team_id",
    "request.api_key.user_id",
    "model.id",
    "model.name",
    "provider_key.id",
    "provider_key.name",
];

/// Per-request values the [`HEADER_TEMPLATE_VARS`] resolve against.
///
/// Every field is optional because the underlying resource field is
/// optional (`team_id` on a personal key) or because the call site has no
/// such resource in scope. An absent field makes any template naming it
/// unresolvable — see the module docs.
#[derive(Debug, Clone, Copy, Default)]
pub struct HeaderVars<'a> {
    pub request_id: Option<&'a str>,
    pub api_key_id: Option<&'a str>,
    pub api_key_name: Option<&'a str>,
    pub api_key_team_id: Option<&'a str>,
    pub api_key_user_id: Option<&'a str>,
    pub model_id: Option<&'a str>,
    pub model_name: Option<&'a str>,
    pub provider_key_id: Option<&'a str>,
    pub provider_key_name: Option<&'a str>,
}

impl<'a> HeaderVars<'a> {
    /// Resolve one variable name. `None` for both an unknown name and a
    /// known-but-absent value — the caller treats them identically
    /// (the header is dropped), and cp-api has already rejected the
    /// unknown-name case at write time.
    fn resolve(&self, name: &str) -> Option<&'a str> {
        let v = match name {
            "request.id" => self.request_id,
            "request.api_key.id" => self.api_key_id,
            "request.api_key.name" => self.api_key_name,
            "request.api_key.team_id" => self.api_key_team_id,
            "request.api_key.user_id" => self.api_key_user_id,
            "model.id" => self.model_id,
            "model.name" => self.model_name,
            "provider_key.id" => self.provider_key_id,
            "provider_key.name" => self.provider_key_name,
            _ => None,
        }?;
        // An empty string is the same "no value for this request" state as
        // an absent field — see the module docs on why a blank header is
        // worse than no header.
        (!v.is_empty()).then_some(v)
    }
}

/// True when `value` contains at least one `${...}` reference.
///
/// Lets the apply path skip rendering entirely for the overwhelmingly
/// common static-header case.
pub fn is_template(value: &str) -> bool {
    value.contains("${")
}

/// Render a `default_headers` value, substituting every `${name}` for its
/// per-request value.
///
/// Returns `None` when the template cannot be fully resolved — an unknown
/// variable name, a variable with no value for this request, or an
/// unterminated `${`. The caller drops that header rather than sending a
/// partially-substituted value. A value with no `${` renders to itself.
///
/// A resolved value that would inject CR/LF/NUL is rejected the same way
/// (`None`): the substituted-in data is a display name an operator typed
/// into the dashboard, so it is not trusted to be header-safe even though
/// cp-api validates the template's own literal text.
pub fn render_header_template(value: &str, vars: &HeaderVars<'_>) -> Option<String> {
    if !is_template(value) {
        return Some(value.to_string());
    }
    let mut out = String::with_capacity(value.len());
    let mut rest = value;
    while let Some(start) = rest.find("${") {
        out.push_str(&rest[..start]);
        let after = &rest[start + 2..];
        let end = after.find('}')?;
        let resolved = vars.resolve(after[..end].trim())?;
        // `write!` into a String is infallible.
        let _ = out.write_str(resolved);
        rest = &after[end + 1..];
    }
    out.push_str(rest);
    if out.contains(['\r', '\n', '\0']) {
        return None;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vars() -> HeaderVars<'static> {
        HeaderVars {
            request_id: Some("req-1"),
            api_key_id: Some("ak-uuid"),
            api_key_name: Some("acme-prod"),
            api_key_team_id: Some("team-7"),
            api_key_user_id: Some("user-9"),
            model_id: Some("m-uuid"),
            model_name: Some("gpt-4o"),
            provider_key_id: Some("pk-uuid"),
            provider_key_name: Some("openai-main"),
        }
    }

    #[test]
    fn static_value_renders_to_itself() {
        assert_eq!(
            render_header_template("plain", &vars()).as_deref(),
            Some("plain")
        );
        assert!(!is_template("plain"));
    }

    #[test]
    fn every_documented_variable_resolves() {
        for name in HEADER_TEMPLATE_VARS {
            let rendered = render_header_template(&format!("${{{name}}}"), &vars());
            assert!(
                rendered.is_some_and(|v| !v.is_empty()),
                "documented variable {name} must resolve"
            );
        }
    }

    #[test]
    fn mixed_literal_and_multiple_variables() {
        assert_eq!(
            render_header_template("k=${request.api_key.name}/${model.name}!", &vars()).as_deref(),
            Some("k=acme-prod/gpt-4o!")
        );
    }

    #[test]
    fn unknown_variable_is_unresolvable() {
        assert_eq!(render_header_template("${request.secret}", &vars()), None);
        // cp-api rejects these at write time; belt-and-braces at runtime.
        assert_eq!(render_header_template("${api_key.secret}", &vars()), None);
    }

    #[test]
    fn absent_or_empty_variable_drops_the_header() {
        let no_team = HeaderVars {
            api_key_team_id: None,
            ..vars()
        };
        assert_eq!(
            render_header_template("${request.api_key.team_id}", &no_team),
            None
        );
        let blank_team = HeaderVars {
            api_key_team_id: Some(""),
            ..vars()
        };
        assert_eq!(
            render_header_template("t=${request.api_key.team_id}", &blank_team),
            None,
            "a blank value must drop the header, not send `t=`"
        );
    }

    #[test]
    fn unterminated_reference_is_unresolvable() {
        assert_eq!(render_header_template("${request.id", &vars()), None);
    }

    #[test]
    fn resolved_value_carrying_crlf_is_rejected() {
        let injected = HeaderVars {
            api_key_name: Some("evil\r\nx-admin: 1"),
            ..vars()
        };
        assert_eq!(
            render_header_template("${request.api_key.name}", &injected),
            None
        );
    }

    #[test]
    fn whitespace_inside_the_reference_is_tolerated() {
        assert_eq!(
            render_header_template("${ request.id }", &vars()).as_deref(),
            Some("req-1")
        );
    }
}
