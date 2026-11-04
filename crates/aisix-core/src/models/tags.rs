//! Cost-attribution tags: the `tags` object an `api_key`, `team` or `user`
//! document carries, the rule that folds the three into the one set a request
//! is attributed to, and the text a verified JWT claim yields as a tag value
//! (`OidcProvider::tag_claims`).

use std::collections::BTreeMap;

use serde_json::{json, Value};

/// Tag key → tag value. A `BTreeMap` so every serialization of one set is
/// byte-identical.
pub type Tags = BTreeMap<String, String>;

/// The tag-key grammar, shared by the resource field and the `tag:<key>`
/// rate-limit policy dimension.
pub const TAG_KEY_PATTERN: &str = "^[a-z][a-z0-9_]{0,62}$";
/// Maximum entries in one document's `tags`.
pub const MAX_TAGS: usize = 32;
/// Maximum length of one tag value, in characters.
pub const MAX_TAG_VALUE_CHARS: usize = 128;

/// Whether `key` is a well-formed tag key ([`TAG_KEY_PATTERN`]).
pub fn is_valid_tag_key(key: &str) -> bool {
    let mut chars = key.chars();
    chars.next().is_some_and(|c| c.is_ascii_lowercase())
        && key.len() <= 63
        && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
}

/// The tags a request takes from documents, resolved per key with the nearest
/// document winning: the API key's own value, else the value of the team its
/// `team_id` names, else the value of the member its `user_id` names. One
/// value per key — never a union.
///
/// A request authenticated by a JWT goes on to override these per key with
/// what its provider's `tag_claims` read from the token; `AuthenticatedKey::tags`
/// is the final set, and the one everything on the request path reads.
pub fn effective_tags<'a>(key: &'a Tags, team: Option<&'a Tags>, user: Option<&'a Tags>) -> Tags {
    let mut out = Tags::new();
    for layer in [user, team, Some(key)].into_iter().flatten() {
        for (k, v) in layer {
            out.insert(k.clone(), v.clone());
        }
    }
    out
}

/// The tag value a JWT claim yields, or why it yields none, in words fit for a
/// log line. The claim value itself never appears in the reason.
///
/// A string is used as is and a number as its JSON text. An array of strings
/// and numbers becomes its distinct texts, sorted and joined with commas, so
/// one set of values is one tag value whatever order the identity provider
/// lists it in. Each text, and the joined result, must pass the same rule the
/// control plane applies to a tag value written to a document (`ValidValue` in
/// its `tags` package): it is rejected, never truncated or trimmed.
pub fn tag_value_from_claim(claim: &Value) -> Result<String, &'static str> {
    fn scalar_text(value: &Value) -> Option<String> {
        match value {
            Value::String(s) => Some(s.clone()),
            Value::Number(n) => Some(n.to_string()),
            _ => None,
        }
    }
    fn check(text: &str) -> Result<(), &'static str> {
        let chars = text.chars().count();
        if chars == 0 || chars > MAX_TAG_VALUE_CHARS {
            return Err("length");
        }
        if text.chars().any(char::is_control) {
            return Err("control_character");
        }
        if text.starts_with(char::is_whitespace) || text.ends_with(char::is_whitespace) {
            return Err("surrounding_whitespace");
        }
        Ok(())
    }
    let Value::Array(items) = claim else {
        let text = scalar_text(claim).ok_or("value_type")?;
        check(&text)?;
        return Ok(text);
    };
    if items.is_empty() {
        return Err("empty_array");
    }
    let mut texts = items
        .iter()
        .map(scalar_text)
        .collect::<Option<Vec<_>>>()
        .ok_or("array_element_type")?;
    texts.iter().try_for_each(|text| check(text))?;
    texts.sort_unstable();
    texts.dedup();
    let joined = texts.join(",");
    check(&joined)?;
    Ok(joined)
}

/// The schema `schemars` renders for a `tags` field: an object of strings,
/// or `null` read as no tags ([`null_as_empty`]). This is the read shape;
/// [`constrain_tags_property`] narrows it to the write-path contract — an
/// object within the limits — on the strict schema only.
pub(crate) fn tags_schema(_: &mut schemars::gen::SchemaGenerator) -> schemars::schema::Schema {
    serde_json::from_value(json!({
        "type": ["object", "null"],
        "additionalProperties": { "type": "string" },
    }))
    .expect("tags schema is a valid schema")
}

/// Deserialize a `tags` field, reading `null` as no tags: a stored row
/// carrying it must not be skipped, which for an `api_key` would stop the
/// key authenticating.
pub(crate) fn null_as_empty<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Tags, D::Error> {
    use serde::Deserialize;
    Ok(Option::<Tags>::deserialize(d)?.unwrap_or_default())
}

/// Add the key grammar, value length and entry cap to a strict root
/// schema's `tags` property.
///
/// Write path only, like the other limits a stored `api_key` row could
/// otherwise be lost to: the loader skips a row it rejects whole, so a tag
/// limit a later release relaxes must not stop an older gateway from
/// authenticating the key that carries it.
pub(crate) fn constrain_tags_property(schema: &mut Value) {
    constrain_tag_keyed_property(
        schema,
        "/properties/tags",
        json!({ "type": "string", "minLength": 1, "maxLength": MAX_TAG_VALUE_CHARS }),
    );
}

/// Add the key grammar, non-empty paths and entry cap to a strict root
/// schema's `tag_claims` property. Write path only, for the same reason as
/// [`constrain_tags_property`]: a stored `oidc_provider` row it dropped would
/// stop authenticating every caller of that provider.
pub(crate) fn constrain_tag_claims_property(schema: &mut Value) {
    constrain_tag_keyed_property(
        schema,
        "/properties/tag_claims",
        json!({ "type": "string", "minLength": 1 }),
    );
}

fn constrain_tag_keyed_property(schema: &mut Value, pointer: &str, values: Value) {
    let Some(property) = schema.pointer_mut(pointer).and_then(Value::as_object_mut) else {
        return;
    };
    property.insert("type".into(), json!("object"));
    property.insert("maxProperties".into(), json!(MAX_TAGS));
    property.insert(
        "propertyNames".into(),
        json!({ "pattern": TAG_KEY_PATTERN }),
    );
    property.insert("additionalProperties".into(), values);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tags(pairs: &[(&str, &str)]) -> Tags {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn nearest_document_wins_per_key_and_keys_combine() {
        let key = tags(&[("department", "key-dept")]);
        let team = tags(&[("department", "team-dept"), ("bg", "team-bg")]);
        let user = tags(&[
            ("department", "user-dept"),
            ("bg", "user-bg"),
            ("subsidiary", "s1"),
        ]);
        assert_eq!(
            effective_tags(&key, Some(&team), Some(&user)),
            tags(&[
                ("bg", "team-bg"),
                ("department", "key-dept"),
                ("subsidiary", "s1")
            ]),
        );
        assert_eq!(effective_tags(&Tags::new(), None, Some(&user)), user,);
        assert!(effective_tags(&Tags::new(), None, None).is_empty());
    }

    #[test]
    fn null_tags_read_as_none_on_every_tagged_document() {
        let key: crate::models::ApiKey =
            serde_json::from_value(json!({"key_hash": "h", "tags": null})).unwrap();
        assert!(key.tags.is_empty());
        let team: crate::models::Team =
            serde_json::from_value(json!({"name": "t", "tags": null})).unwrap();
        assert!(team.tags.is_empty());
        let user: crate::models::User =
            serde_json::from_value(json!({"name": "u", "tags": null})).unwrap();
        assert!(user.tags.is_empty());
        let null = json!({"key_hash": "h", "tags": null});
        assert!(crate::models::validate_apikey_lenient(&null).is_ok());
        assert!(crate::models::validate_apikey(&null).is_err());
    }

    #[test]
    fn claim_values_become_tag_text() {
        for (claim, want) in [
            (json!("rnd"), "rnd"),
            (json!("研发 一部"), "研发 一部"),
            (json!(42), "42"),
            (json!(1.5), "1.5"),
            (json!(["rnd"]), "rnd"),
            (json!([7]), "7"),
            (json!(["b", "a", "b"]), "a,b"),
            (json!([10, "9", 10]), "10,9"),
            (json!(["1", 1]), "1"),
            (json!(["a,b", "c"]), "a,b,c"),
            (
                json!("x".repeat(MAX_TAG_VALUE_CHARS)),
                &*"x".repeat(MAX_TAG_VALUE_CHARS),
            ),
            (
                json!("字".repeat(MAX_TAG_VALUE_CHARS)),
                &*"字".repeat(MAX_TAG_VALUE_CHARS),
            ),
        ] {
            assert_eq!(tag_value_from_claim(&claim).as_deref(), Ok(want), "{claim}");
        }
    }

    #[test]
    fn claim_values_that_yield_no_tag() {
        for (claim, reason) in [
            (json!(true), "value_type"),
            (Value::Null, "value_type"),
            (json!({"a": 1}), "value_type"),
            (json!([]), "empty_array"),
            (json!(["a", true]), "array_element_type"),
            (json!(["a", null]), "array_element_type"),
            (json!([["a"]]), "array_element_type"),
            (json!(""), "length"),
            (json!([""]), "length"),
            (json!("x".repeat(MAX_TAG_VALUE_CHARS + 1)), "length"),
            (json!(["x".repeat(100), "y".repeat(100)]), "length"),
            (json!(["", "rnd"]), "length"),
            (
                json!(["x".repeat(MAX_TAG_VALUE_CHARS + 1), "rnd"]),
                "length",
            ),
            (json!(["a ", "b"]), "surrounding_whitespace"),
            (json!(["b", " a"]), "surrounding_whitespace"),
            (json!(["a\nb", "c"]), "control_character"),
            (json!("a\nb"), "control_character"),
            (json!("a\u{7f}b"), "control_character"),
            (json!(" rnd"), "surrounding_whitespace"),
            (json!("rnd "), "surrounding_whitespace"),
            (json!("rnd\u{3000}"), "surrounding_whitespace"),
        ] {
            assert_eq!(tag_value_from_claim(&claim), Err(reason), "{claim}");
        }
    }

    #[test]
    fn tag_key_grammar() {
        for ok in ["a", "department", "cost_center_2", &"a".repeat(63)] {
            assert!(is_valid_tag_key(ok), "{ok}");
        }
        for bad in [
            "",
            "Department",
            "1dept",
            "_x",
            "dept-x",
            "dépt",
            &"a".repeat(64),
        ] {
            assert!(!is_valid_tag_key(bad), "{bad}");
        }
    }
}
