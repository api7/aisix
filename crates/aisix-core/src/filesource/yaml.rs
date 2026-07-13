//! YAML tree → JSON document conversion with `${VAR}` interpolation.
//!
//! Interpolation runs on string scalars in the *parsed* YAML tree —
//! after the YAML parse, before the JSON conversion — so an environment
//! value can never inject YAML structure. Mapping keys are deliberately
//! not interpolated for the same reason.
//!
//! Interpolation grammar (normative for the resources file):
//! - `${VAR}`  → the value of environment variable `VAR`; partial
//!   interpolation is supported (`https://${HOST}/v1`). A missing or
//!   empty variable is an error.
//! - `$$`      → a literal `$`.
//! - bare `$VAR` (no braces) is NOT interpolated and passes through.

use serde_json::{Map, Number, Value};
use yaml_rust2::Yaml;

/// Environment lookup used by interpolation. Production passes
/// `std::env::var(..).ok()`; tests inject a closed map so they never
/// mutate process-global env state.
pub(crate) type EnvLookup<'a> = &'a dyn Fn(&str) -> Option<String>;

/// Interpolate `${VAR}` references in one string scalar.
pub(crate) fn interpolate_str(input: &str, env: EnvLookup<'_>) -> Result<String, String> {
    let mut out = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '$' {
            out.push(c);
            continue;
        }
        match chars.peek() {
            // `$$` → literal `$`.
            Some('$') => {
                chars.next();
                out.push('$');
            }
            // `${VAR}` → env lookup.
            Some('{') => {
                chars.next();
                let mut name = String::new();
                let mut closed = false;
                for c2 in chars.by_ref() {
                    if c2 == '}' {
                        closed = true;
                        break;
                    }
                    name.push(c2);
                }
                if !closed {
                    return Err(format!(
                        "unterminated `${{` in value {input:?} — close it with `}}` \
                         or escape the dollar sign as `$$`"
                    ));
                }
                if name.is_empty() {
                    return Err("empty variable name in `${}`".into());
                }
                match env(&name) {
                    Some(v) if !v.is_empty() => out.push_str(&v),
                    _ => {
                        return Err(format!("environment variable `{name}` is unset or empty"));
                    }
                }
            }
            // Bare `$VAR` (or trailing `$`) — not an interpolation form.
            _ => out.push('$'),
        }
    }
    Ok(out)
}

/// Convert a parsed YAML node into a `serde_json::Value`, interpolating
/// `${VAR}` in every string scalar along the way. Errors carry the
/// field path relative to the entry root (e.g. `routing.targets[0].model`).
pub(crate) fn yaml_to_json(
    node: &Yaml,
    path: &str,
    env: EnvLookup<'_>,
) -> Result<Value, (String, String)> {
    let at = |msg: String| (path.to_string(), msg);
    match node {
        Yaml::Null => Ok(Value::Null),
        Yaml::Boolean(b) => Ok(Value::Bool(*b)),
        Yaml::Integer(i) => Ok(Value::Number(Number::from(*i))),
        Yaml::Real(raw) => {
            let f: f64 = raw
                .parse()
                .map_err(|_| at(format!("invalid number {raw:?}")))?;
            Number::from_f64(f)
                .map(Value::Number)
                .ok_or_else(|| at(format!("number {raw:?} is not representable in JSON")))
        }
        Yaml::String(s) => interpolate_str(s, env).map(Value::String).map_err(at),
        Yaml::Array(items) => {
            let mut out = Vec::with_capacity(items.len());
            for (i, item) in items.iter().enumerate() {
                let child_path = format!("{path}[{i}]");
                out.push(yaml_to_json(item, &child_path, env)?);
            }
            Ok(Value::Array(out))
        }
        Yaml::Hash(map) => {
            let mut out = Map::with_capacity(map.len());
            for (k, v) in map {
                let key = match k {
                    Yaml::String(s) => s.clone(),
                    other => {
                        return Err(at(format!(
                            "mapping keys must be plain strings, found {other:?}"
                        )));
                    }
                };
                let child_path = if path.is_empty() {
                    key.clone()
                } else {
                    format!("{path}.{key}")
                };
                out.insert(key, yaml_to_json(v, &child_path, env)?);
            }
            Ok(Value::Object(out))
        }
        Yaml::Alias(_) | Yaml::BadValue => {
            Err(at("unsupported YAML construct at this position".into()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use yaml_rust2::YamlLoader;

    fn env_of(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    fn lookup(map: &HashMap<String, String>) -> impl Fn(&str) -> Option<String> + '_ {
        move |name| map.get(name).cloned()
    }

    #[test]
    fn interpolates_full_and_partial_values() {
        let env = env_of(&[("HOST", "up.example"), ("KEY", "sk-1")]);
        let f = lookup(&env);
        assert_eq!(interpolate_str("${KEY}", &f).unwrap(), "sk-1");
        assert_eq!(
            interpolate_str("https://${HOST}/v1", &f).unwrap(),
            "https://up.example/v1"
        );
        assert_eq!(
            interpolate_str("${HOST}:${KEY}", &f).unwrap(),
            "up.example:sk-1"
        );
    }

    #[test]
    fn double_dollar_escapes_a_literal_dollar() {
        let env = env_of(&[("VAR", "x")]);
        let f = lookup(&env);
        assert_eq!(interpolate_str("$${VAR}", &f).unwrap(), "${VAR}");
        assert_eq!(interpolate_str("a$$b", &f).unwrap(), "a$b");
        assert_eq!(interpolate_str("$$$$", &f).unwrap(), "$$");
        // `$$` then a real interpolation.
        assert_eq!(interpolate_str("$$${VAR}", &f).unwrap(), "$x");
    }

    #[test]
    fn bare_dollar_var_is_left_untouched() {
        let env = env_of(&[("VAR", "x")]);
        let f = lookup(&env);
        assert_eq!(interpolate_str("$VAR", &f).unwrap(), "$VAR");
        assert_eq!(interpolate_str("cost is 5$", &f).unwrap(), "cost is 5$");
        assert_eq!(interpolate_str("a$ b", &f).unwrap(), "a$ b");
    }

    #[test]
    fn missing_or_empty_variable_is_an_error() {
        let env = env_of(&[("EMPTY", "")]);
        let f = lookup(&env);
        let err = interpolate_str("${NOPE}", &f).unwrap_err();
        assert!(err.contains("`NOPE`"), "unexpected: {err}");
        assert!(err.contains("unset or empty"), "unexpected: {err}");
        let err = interpolate_str("${EMPTY}", &f).unwrap_err();
        assert!(err.contains("`EMPTY`"), "unexpected: {err}");
    }

    #[test]
    fn unterminated_and_empty_braces_are_errors() {
        let env = env_of(&[]);
        let f = lookup(&env);
        assert!(interpolate_str("${OPEN", &f)
            .unwrap_err()
            .contains("unterminated"));
        assert!(interpolate_str("${}", &f)
            .unwrap_err()
            .contains("empty variable name"));
    }

    #[test]
    fn conversion_interpolates_only_string_scalars_and_tracks_paths() {
        let env = env_of(&[("HOST", "h")]);
        let f = lookup(&env);
        let docs = YamlLoader::load_from_str(
            "api_base: https://${HOST}/v1\nnested:\n  list:\n    - ${MISSING}\n",
        )
        .unwrap();

        // Happy path on the string scalar.
        let ok = yaml_to_json(&docs[0]["api_base"], "api_base", &f).unwrap();
        assert_eq!(ok, serde_json::json!("https://h/v1"));

        // Error path carries the full field path.
        let (path, msg) = yaml_to_json(&docs[0], "", &f).unwrap_err();
        assert_eq!(path, "nested.list[0]");
        assert!(msg.contains("`MISSING`"));
    }

    #[test]
    fn conversion_preserves_scalar_types() {
        let env = env_of(&[]);
        let f = lookup(&env);
        let docs = YamlLoader::load_from_str("i: 3\nf: 1.5\nb: true\nn: null\ns: hi\n").unwrap();
        let v = yaml_to_json(&docs[0], "", &f).unwrap();
        assert_eq!(
            v,
            serde_json::json!({"i": 3, "f": 1.5, "b": true, "n": null, "s": "hi"})
        );
    }

    #[test]
    fn conversion_rejects_non_string_mapping_keys() {
        let env = env_of(&[]);
        let f = lookup(&env);
        let docs = YamlLoader::load_from_str("1: x\n").unwrap();
        let (_, msg) = yaml_to_json(&docs[0], "", &f).unwrap_err();
        assert!(msg.contains("plain strings"), "unexpected: {msg}");
    }
}
