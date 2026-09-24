//! OpenAPI-to-MCP tool generation for `type: openapi` MCP servers.
//!
//! Each `paths` operation becomes one tool: the tool name is the sanitized
//! `operationId` (fallback `<method>_<path>`), path/query parameters become
//! top-level schema properties, and a JSON request body becomes a single
//! `body` property. Local `$ref`s are resolved (bounded) so referenced
//! schemas keep their shape.
//!
//! This lives in the core crate, not beside the bridge that executes the
//! tools, so the resource loaders (file source, etcd, `aisix validate`) judge
//! a document with the very walk the runtime serves tools from.

use std::collections::HashSet;

use serde_json::{json, Map, Value};

/// Tool names must survive every major LLM provider's `^[a-zA-Z0-9_-]+$`
/// name check; 128 is the most restrictive cap.
pub const TOOL_NAME_MAX_LEN: usize = 128;

/// HTTP methods that map to tools, in generation order.
const METHODS: [&str; 5] = ["get", "post", "put", "delete", "patch"];

/// `$ref` inlining bounds, per operation: a cyclic or pathologically nested
/// schema degrades to `{}` (schema "anything") instead of recursing forever
/// or exploding the inlined size.
const MAX_REF_DEPTH: usize = 16;
const MAX_REF_EXPANSIONS: usize = 256;

/// One OpenAPI operation, resolved into a callable tool.
#[derive(Debug, Clone, PartialEq)]
pub struct GeneratedTool {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
    /// Lowercase HTTP method.
    pub method: String,
    /// The path template as written in the spec, e.g. `/items/{id}`.
    pub path: String,
    pub path_params: Vec<String>,
    pub query_params: Vec<String>,
    pub has_body: bool,
}

/// Map an `operationId` (or fallback) to a provider-safe tool name:
/// lowercase, any character outside `[a-zA-Z0-9_-]` replaced with `_`,
/// capped at [`TOOL_NAME_MAX_LEN`].
pub fn sanitize_tool_name(raw: &str) -> String {
    raw.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .take(TOOL_NAME_MAX_LEN)
        .collect()
}

/// The load-time verdict on a `type: openapi` server's document: an error
/// when the gateway would serve no tools from it — no usable `paths`, or no
/// operation it can turn into a tool. Collisions are not an error: they
/// load, and every colliding operation stays callable under its suffix.
pub fn check_spec_yields_tools(spec: &Value) -> Result<(), String> {
    let generation = generate(spec)?;
    if !generation.tools.is_empty() {
        return Ok(());
    }
    let mut message =
        "openapi spec has no operations that can become tools (methods get/post/put/delete/patch)"
            .to_string();
    if !generation.skipped.is_empty() {
        message.push_str(&format!(
            "; skipped operations without an application/json request body: {}",
            generation.skipped.join(", ")
        ));
    }
    Err(message)
}

/// Outcome of walking a spec: the tools plus the anomalies a load or
/// `aisix validate` reports.
pub struct Generation {
    pub tools: Vec<GeneratedTool>,
    /// Base names that collided after sanitization (each listed once).
    /// Every colliding operation is still generated, under a `_2` / `_3` …
    /// suffix.
    pub duplicates: Vec<String>,
    /// `<METHOD> <path>` of operations skipped for an unsupported body.
    pub skipped: Vec<String>,
}

/// Generate the tool set from an OpenAPI 3.x document.
///
/// Anomalies inside a single operation (a request body without an
/// `application/json` variant, an unresolvable parameter ref) skip or degrade
/// that operation only; the error path is reserved for a document that has no
/// usable `paths` object at all.
pub fn generate(spec: &Value) -> Result<Generation, String> {
    let paths = spec
        .get("paths")
        .and_then(Value::as_object)
        .ok_or_else(|| "openapi spec has no `paths` object".to_string())?;

    let components = spec.get("components").cloned().unwrap_or(Value::Null);
    let mut used_names: HashSet<String> = HashSet::new();
    let mut tools = Vec::new();
    let mut duplicates: Vec<String> = Vec::new();
    let mut skipped: Vec<String> = Vec::new();

    for (path, path_item) in paths {
        let Some(path_item) = path_item.as_object() else {
            continue;
        };
        for method in METHODS {
            let Some(operation) = path_item.get(method).and_then(Value::as_object) else {
                continue;
            };

            let mut resolver = RefResolver::new(spec);
            let params = merged_parameters(path_item, operation, &components, &mut resolver);

            // A request body we cannot express (no JSON variant) skips the
            // operation: a tool missing its body argument would mislead the
            // agent into calls that cannot succeed.
            let request_body = operation
                .get("requestBody")
                .map(|rb| resolver.resolve(rb, 0));
            let body_schema = match &request_body {
                Some(rb) => match json_body_schema(rb, &mut resolver) {
                    BodyOutcome::Schema(schema) => Some(schema),
                    BodyOutcome::None => None,
                    BodyOutcome::Unsupported => {
                        skipped.push(format!("{} {}", method.to_uppercase(), path));
                        continue;
                    }
                },
                None => None,
            };

            let raw_name = operation
                .get("operationId")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .unwrap_or_else(|| format!("{method}_{path}"));
            let base_name = sanitize_tool_name(&raw_name);

            // Disambiguate names that collide after sanitization so every
            // tool stays reachable (`foo/list` and `foo.list` both map to
            // `foo_list`).
            let mut name = base_name.clone();
            let mut n = 1;
            while !used_names.insert(name.clone()) {
                if n == 1 && !duplicates.contains(&base_name) {
                    duplicates.push(base_name.clone());
                }
                n += 1;
                let suffix = format!("_{n}");
                let keep = TOOL_NAME_MAX_LEN - suffix.len();
                name = base_name.chars().take(keep).collect::<String>() + &suffix;
            }

            let description = operation
                .get("summary")
                .or_else(|| operation.get("description"))
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .unwrap_or_else(|| format!("{} {}", method.to_uppercase(), path));

            let mut properties = Map::new();
            let mut required = Vec::new();
            let mut path_params = Vec::new();
            let mut query_params = Vec::new();

            for param in &params {
                let Some(param_name) = param.get("name").and_then(Value::as_str) else {
                    continue;
                };
                let location = param.get("in").and_then(Value::as_str).unwrap_or("");
                match location {
                    "path" => path_params.push(param_name.to_string()),
                    "query" => query_params.push(param_name.to_string()),
                    // header/cookie parameters are not exposed to the agent:
                    // upstream headers are the gateway's to set, not the
                    // caller's.
                    _ => continue,
                }

                let mut schema = match param.get("schema") {
                    Some(s) => resolver.resolve(s, 0),
                    None => Value::Null,
                };
                if !schema.is_object() || schema.as_object().is_some_and(Map::is_empty) {
                    schema = json!({ "type": "string" });
                }
                if let Some(desc) = param.get("description").and_then(Value::as_str) {
                    schema
                        .as_object_mut()
                        .expect("schema coerced to object above")
                        .insert("description".to_string(), json!(desc));
                }
                properties.insert(param_name.to_string(), schema);

                if param.get("required").and_then(Value::as_bool) == Some(true) {
                    required.push(param_name.to_string());
                }
            }

            let has_body = if let Some(mut schema) = body_schema {
                // A non-object or degraded-to-`{}` schema still gets an
                // object hint, mirroring the string default on parameters.
                if !schema.is_object() || schema.as_object().is_some_and(Map::is_empty) {
                    schema = json!({ "type": "object" });
                }
                if let Some(desc) = request_body
                    .as_ref()
                    .and_then(|rb| rb.get("description"))
                    .and_then(Value::as_str)
                {
                    schema
                        .as_object_mut()
                        .expect("schema coerced to object above")
                        .insert("description".to_string(), json!(desc));
                }
                properties.insert("body".to_string(), schema);
                if request_body
                    .as_ref()
                    .and_then(|rb| rb.get("required"))
                    .and_then(Value::as_bool)
                    == Some(true)
                {
                    required.push("body".to_string());
                }
                true
            } else {
                false
            };

            tools.push(GeneratedTool {
                name,
                description,
                input_schema: json!({
                    "type": "object",
                    "properties": properties,
                    "required": required,
                }),
                method: method.to_string(),
                path: path.clone(),
                path_params,
                query_params,
                has_body,
            });
        }
    }

    Ok(Generation {
        tools,
        duplicates,
        skipped,
    })
}

/// How an operation's `requestBody` maps onto the tool schema.
enum BodyOutcome {
    /// `application/json` variant found; its (resolved) schema.
    Schema(Value),
    /// The body object carries no `content` at all — treat as body-less.
    None,
    /// A body exists but has no JSON variant (multipart upload, form data…).
    Unsupported,
}

fn json_body_schema(request_body: &Value, resolver: &mut RefResolver<'_>) -> BodyOutcome {
    let Some(content) = request_body.get("content").and_then(Value::as_object) else {
        return BodyOutcome::None;
    };
    if content.is_empty() {
        return BodyOutcome::None;
    }
    // Accept `application/json` plus parameterized variants like
    // `application/json; charset=utf-8` or `application/problem+json`.
    let json_variant = content.iter().find(|(mime, _)| {
        let mime = mime.split(';').next().unwrap_or("").trim();
        mime.eq_ignore_ascii_case("application/json")
            || (mime.starts_with("application/") && mime.ends_with("+json"))
    });
    match json_variant {
        Some((_, media)) => {
            let schema = media
                .get("schema")
                .map(|s| resolver.resolve(s, 0))
                .unwrap_or_else(|| json!({ "type": "object" }));
            BodyOutcome::Schema(schema)
        }
        None => BodyOutcome::Unsupported,
    }
}

/// Merge path-level and operation-level parameters (operation wins on the
/// same `(name, in)` pair), resolving `#/components/parameters/*` refs and
/// dropping unresolvable entries.
fn merged_parameters(
    path_item: &Map<String, Value>,
    operation: &Map<String, Value>,
    components: &Value,
    resolver: &mut RefResolver<'_>,
) -> Vec<Value> {
    let resolve_list = |raw: Option<&Value>, resolver: &mut RefResolver<'_>| -> Vec<Value> {
        raw.and_then(Value::as_array)
            .map(|list| {
                list.iter()
                    .filter_map(|p| {
                        let resolved = resolve_parameter(p, components, resolver)?;
                        resolved.get("name")?.as_str()?;
                        Some(resolved)
                    })
                    .collect()
            })
            .unwrap_or_default()
    };

    let path_level = resolve_list(path_item.get("parameters"), resolver);
    let op_level = resolve_list(operation.get("parameters"), resolver);

    let op_keys: HashSet<(String, String)> = op_level.iter().map(param_key).collect();
    let mut merged: Vec<Value> = path_level
        .into_iter()
        .filter(|p| !op_keys.contains(&param_key(p)))
        .collect();
    merged.extend(op_level);
    merged
}

fn param_key(param: &Value) -> (String, String) {
    (
        param
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        param
            .get("in")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
    )
}

/// Resolve one parameter entry, following a `#/components/parameters/<name>`
/// ref if present. Returns `None` for unresolvable refs so callers drop the
/// entry instead of keeping a nameless stub.
fn resolve_parameter(
    param: &Value,
    components: &Value,
    resolver: &mut RefResolver<'_>,
) -> Option<Value> {
    let Some(reference) = param.get("$ref").and_then(Value::as_str) else {
        return Some(param.clone());
    };
    let target_name = reference.strip_prefix("#/components/parameters/")?;
    let target = components.get("parameters")?.get(target_name)?;
    Some(resolver.resolve(target, 0))
}

/// Bounded local-`$ref` inliner.
///
/// Replaces `{"$ref": "#/..."}` nodes with their (recursively resolved)
/// targets; sibling keys next to `$ref` overlay the resolved target (the
/// OpenAPI 3.1 `summary`/`description` pattern). External refs, missing
/// targets, cycles past [`MAX_REF_DEPTH`], and documents spending more than
/// [`MAX_REF_EXPANSIONS`] lookups degrade the node to `{}` — schema
/// "anything" — rather than failing the operation.
struct RefResolver<'a> {
    root: &'a Value,
    expansions: usize,
}

impl<'a> RefResolver<'a> {
    fn new(root: &'a Value) -> Self {
        Self {
            root,
            expansions: 0,
        }
    }

    fn resolve(&mut self, node: &Value, depth: usize) -> Value {
        match node {
            Value::Object(map) => {
                if let Some(reference) = map.get("$ref").and_then(Value::as_str) {
                    let resolved = self.resolve_ref(reference, depth);
                    // Sibling keys overlay the resolved target.
                    if map.len() > 1 {
                        let mut base = match resolved {
                            Value::Object(m) => m,
                            _ => Map::new(),
                        };
                        for (k, v) in map {
                            if k != "$ref" {
                                base.insert(k.clone(), self.resolve(v, depth));
                            }
                        }
                        return Value::Object(base);
                    }
                    return resolved;
                }
                Value::Object(
                    map.iter()
                        .map(|(k, v)| (k.clone(), self.resolve(v, depth)))
                        .collect(),
                )
            }
            Value::Array(items) => {
                Value::Array(items.iter().map(|v| self.resolve(v, depth)).collect())
            }
            other => other.clone(),
        }
    }

    fn resolve_ref(&mut self, reference: &str, depth: usize) -> Value {
        if depth >= MAX_REF_DEPTH || self.expansions >= MAX_REF_EXPANSIONS {
            return json!({});
        }
        let Some(pointer) = reference.strip_prefix('#') else {
            // External refs are not fetched at runtime by design.
            return json!({});
        };
        self.expansions += 1;
        match self.root.pointer(pointer) {
            Some(target) => self.resolve(target, depth + 1),
            None => json!({}),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitizes_operation_ids_into_provider_safe_names() {
        // GitHub-style tag-namespaced ids gain `_` for `/`; uppercase folds.
        assert_eq!(
            sanitize_tool_name("actions/Download-Job.Logs"),
            "actions_download-job_logs"
        );
        let long = "x".repeat(200);
        assert_eq!(sanitize_tool_name(&long).len(), TOOL_NAME_MAX_LEN);
    }

    #[test]
    fn a_spec_is_refused_only_when_it_yields_no_tool() {
        // No paths at all.
        assert!(check_spec_yields_tools(&json!({ "openapi": "3.0.0" })).is_err());

        // Paths but nothing generatable — the skipped multipart op is named.
        let only_multipart = json!({
            "paths": { "/upload": { "post": {
                "operationId": "up",
                "requestBody": { "content": { "multipart/form-data": {} } }
            } } }
        });
        let err = check_spec_yields_tools(&only_multipart).unwrap_err();
        assert!(err.contains("no operations"), "{err}");
        assert!(err.contains("POST /upload"), "{err}");

        // A collision is reported, not refused: both operations are served.
        let dup = json!({
            "paths": {
                "/a": { "get": { "operationId": "foo/list" } },
                "/b": { "get": { "operationId": "foo.list" } }
            }
        });
        check_spec_yields_tools(&dup).unwrap();
        let generation = generate(&dup).unwrap();
        assert_eq!(generation.duplicates, vec!["foo_list"]);
        let names: Vec<_> = generation.tools.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, ["foo_list", "foo_list_2"]);

        check_spec_yields_tools(&json!({
            "paths": { "/items": { "get": { "operationId": "listItems" } } }
        }))
        .unwrap();
    }
}
