//! Render an [`ExportDocument`] as a resources-file YAML string.
//!
//! Emission uses `yaml-rust2`'s own `YamlEmitter` — the same crate whose
//! `YamlLoader` the file source parses with — so the output is guaranteed
//! to re-parse to an equivalent tree. In particular the emitter quotes
//! scalars that would otherwise change type (the mandatory
//! `_format_version: "1"` re-parses as the string `"1"`, not the integer
//! `1`), and quotes the `${VAR}` secret placeholders as needed.
//!
//! The top-level mapping is built directly so the header sits first and
//! collections follow in the file's fixed kind order; per-entry field
//! order is whatever `serde_json` produced (deterministic).

use serde_json::Value;
use yaml_rust2::yaml::Hash;
use yaml_rust2::{Yaml, YamlEmitter};

use super::document::ExportDocument;

/// The mandatory format-version header value, emitted as a quoted string.
const FORMAT_VERSION: &str = "1";

/// Serialize the document to a resources-file YAML string.
pub fn emit_yaml(document: &ExportDocument) -> Result<String, String> {
    let mut root = Hash::new();
    root.insert(
        Yaml::String("_format_version".into()),
        Yaml::String(FORMAT_VERSION.into()),
    );
    for (kind, entries) in &document.collections {
        let array = Yaml::Array(entries.iter().map(json_to_yaml).collect());
        root.insert(Yaml::String((*kind).into()), array);
    }

    let mut out = String::new();
    YamlEmitter::new(&mut out)
        .dump(&Yaml::Hash(root))
        .map_err(|e| format!("YAML emit failed: {e}"))?;
    // The emitter writes a leading `---` document marker but no trailing
    // newline; add one so the file is POSIX-clean.
    out.push('\n');
    Ok(out)
}

/// Convert a JSON value to the `yaml-rust2` tree, preserving object key
/// order. Numbers that exceed `i64` fall back to their textual form as a
/// YAML real, which re-parses to the same number.
fn json_to_yaml(value: &Value) -> Yaml {
    match value {
        Value::Null => Yaml::Null,
        Value::Bool(b) => Yaml::Boolean(*b),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Yaml::Integer(i)
            } else {
                Yaml::Real(n.to_string())
            }
        }
        Value::String(s) => Yaml::String(s.clone()),
        Value::Array(items) => Yaml::Array(items.iter().map(json_to_yaml).collect()),
        Value::Object(map) => {
            let mut hash = Hash::new();
            for (key, child) in map {
                hash.insert(Yaml::String(key.clone()), json_to_yaml(child));
            }
            Yaml::Hash(hash)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use yaml_rust2::YamlLoader;

    fn doc_with(collections: Vec<(&'static str, Vec<Value>)>) -> ExportDocument {
        ExportDocument {
            collections,
            secret_placeholders: Vec::new(),
            warnings: Vec::new(),
            blocking: Vec::new(),
        }
    }

    #[test]
    fn format_version_reparses_as_a_string_not_an_integer() {
        let yaml = emit_yaml(&doc_with(vec![])).unwrap();
        let docs = YamlLoader::load_from_str(&yaml).unwrap();
        assert_eq!(docs.len(), 1, "must be a single document");
        // The loader keys off `Yaml::String("1")`; an unquoted `1` would
        // parse as an integer and be rejected.
        assert_eq!(
            docs[0]["_format_version"],
            Yaml::String("1".into()),
            "emitted:\n{yaml}"
        );
    }

    #[test]
    fn secret_placeholder_scalar_round_trips_intact() {
        let collections = vec![(
            "provider_keys",
            vec![json!({
                "display_name": "openai-prod",
                "api_key": "${AISIXSECRET_PROVIDER_KEY_OPENAI_PROD_API_KEY}"
            })],
        )];
        let yaml = emit_yaml(&doc_with(collections)).unwrap();
        let docs = YamlLoader::load_from_str(&yaml).unwrap();
        let pk = &docs[0]["provider_keys"][0];
        assert_eq!(
            pk["api_key"],
            Yaml::String("${AISIXSECRET_PROVIDER_KEY_OPENAI_PROD_API_KEY}".into()),
            "emitted:\n{yaml}"
        );
        assert_eq!(pk["display_name"], Yaml::String("openai-prod".into()));
    }

    #[test]
    fn scalar_types_and_nesting_survive_a_round_trip() {
        let collections = vec![(
            "rate_limit_policies",
            vec![json!({
                "name": "cap",
                "scope": "model",
                "scope_ref": "gpt-4o",
                "window": "minute",
                "max_requests": 300,
                "enabled": true
            })],
        )];
        let yaml = emit_yaml(&doc_with(collections)).unwrap();
        let docs = YamlLoader::load_from_str(&yaml).unwrap();
        let p = &docs[0]["rate_limit_policies"][0];
        // Integer stays an integer; string stays a string; bool a bool.
        assert_eq!(p["max_requests"], Yaml::Integer(300));
        assert_eq!(p["scope_ref"], Yaml::String("gpt-4o".into()));
        assert_eq!(p["enabled"], Yaml::Boolean(true));
    }

    #[test]
    fn collections_are_emitted_in_the_supplied_order() {
        let yaml = emit_yaml(&doc_with(vec![
            (
                "provider_keys",
                vec![json!({"display_name": "pk", "api_key": "x"})],
            ),
            (
                "models",
                vec![json!({"display_name": "m", "provider_key": "pk"})],
            ),
        ]))
        .unwrap();
        let pk_at = yaml.find("provider_keys:").expect("has provider_keys");
        let models_at = yaml.find("models:").expect("has models");
        assert!(pk_at < models_at, "provider_keys before models:\n{yaml}");
    }
}
