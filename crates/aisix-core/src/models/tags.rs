//! Cost-attribution tags: the `tags` object an `api_key`, `team` or `user`
//! document carries, and the rule that folds the three into the one set a
//! request is attributed to.

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

/// A request's effective tags, resolved per key with the nearest document
/// winning: the API key's own value, else the value of the team its
/// `team_id` names, else the value of the member its `user_id` names. One
/// value per key — never a union.
pub fn effective_tags<'a>(key: &'a Tags, team: Option<&'a Tags>, user: Option<&'a Tags>) -> Tags {
    let mut out = Tags::new();
    for layer in [user, team, Some(key)].into_iter().flatten() {
        for (k, v) in layer {
            out.insert(k.clone(), v.clone());
        }
    }
    out
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
    let Some(tags) = schema
        .pointer_mut("/properties/tags")
        .and_then(Value::as_object_mut)
    else {
        return;
    };
    tags.insert("type".into(), json!("object"));
    tags.insert("maxProperties".into(), json!(MAX_TAGS));
    tags.insert(
        "propertyNames".into(),
        json!({ "pattern": TAG_KEY_PATTERN }),
    );
    tags.insert(
        "additionalProperties".into(),
        json!({ "type": "string", "minLength": 1, "maxLength": MAX_TAG_VALUE_CHARS }),
    );
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
