//! OpenAPI-backed MCP bridge (`type: openapi`).
//!
//! Instead of tunnelling to a real upstream MCP server, this bridge generates
//! the tool surface itself from a registered OpenAPI 3.x document and executes
//! `tools/call` as plain HTTP requests against the API's base URL. Each
//! `paths` operation becomes one tool; the gateway-held credential is injected
//! on every outbound request and is never visible to the calling agent.
//!
//! The generation rules follow LiteLLM's `openapi_to_mcp_generator` so tool
//! names and argument shapes stay familiar across gateways: the tool name is
//! the sanitized `operationId` (fallback `<method>_<path>`), path/query
//! parameters become top-level schema properties, and a JSON request body
//! becomes a single `body` property. Two deliberate improvements over the
//! baseline: local `$ref`s are resolved (bounded) so referenced schemas keep
//! their shape, and a non-2xx response is flagged `is_error` so the agent can
//! react to a failed call.
//!
//! The spec is read from the resource snapshot (shared, never re-fetched at
//! runtime): the control plane validates and materializes it at write time, so
//! the tool set only changes when the resource does.

use std::collections::HashSet;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use aisix_core::{McpAuthType, McpServer, ResourceEntry};
use async_trait::async_trait;
use percent_encoding::{utf8_percent_encode, AsciiSet, CONTROLS};
use serde_json::{json, Map, Value};

use crate::bridge::{McpBridge, McpTool, McpToolResult, OAuthClientConfig};
use crate::error::McpError;

/// Header the API key is sent under for `auth_type: api_key` when the
/// resource sets no `api_key_header`.
pub const DEFAULT_API_KEY_HEADER: &str = "x-api-key";

/// Tool names must survive every major LLM provider's `^[a-zA-Z0-9_-]+$`
/// name check; 128 is the most restrictive cap (mirrors LiteLLM).
const TOOL_NAME_MAX_LEN: usize = 128;

/// HTTP methods that map to tools, in generation order.
const METHODS: [&str; 5] = ["get", "post", "put", "delete", "patch"];

/// `$ref` inlining bounds, per operation: a cyclic or pathologically nested
/// schema degrades to `{}` (schema "anything") instead of recursing forever
/// or exploding the inlined size.
const MAX_REF_DEPTH: usize = 16;
const MAX_REF_EXPANSIONS: usize = 256;

/// Everything except RFC 3986 unreserved characters is percent-encoded when a
/// path parameter value is substituted into the URL template.
const PATH_SEGMENT_ENCODE: &AsciiSet = &CONTROLS
    .add(b' ')
    .add(b'"')
    .add(b'#')
    .add(b'%')
    .add(b'/')
    .add(b'<')
    .add(b'>')
    .add(b'?')
    .add(b'`')
    .add(b'{')
    .add(b'}')
    .add(b'\\')
    .add(b'^')
    .add(b'|')
    .add(b'&')
    .add(b'+')
    .add(b',')
    .add(b':')
    .add(b';')
    .add(b'=')
    .add(b'@')
    .add(b'[')
    .add(b']')
    .add(b'!')
    .add(b'$')
    .add(b'\'')
    .add(b'(')
    .add(b')')
    .add(b'*');

/// Shared HTTP client for generated tool calls: the process-wide upstream
/// connection settings, no redirect following (a redirect could re-send the
/// gateway-held credential to a host the operator never configured).
fn http_client() -> &'static reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        aisix_gateway::client_builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .expect("failed to build openapi tool HTTP client")
    })
}

/// One OpenAPI operation, resolved into a callable tool.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct GeneratedTool {
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

/// [`McpBridge`] over an OpenAPI-backed `mcp_server` resource. Holds the
/// snapshot entry (`Arc`, shared with the snapshot) so the spec is never
/// deep-cloned per request.
pub struct OpenApiBridge {
    entry: Arc<ResourceEntry<McpServer>>,
    timeout: Duration,
    /// Header name and value delivering the calling agent's verified JWT
    /// to the REST API behind these tools, resolved per request by
    /// `aisix_core::forwarded_jwt` from the server's `forward_jwt_header`.
    /// `None` when unconfigured or the agent authenticated with an API key.
    forwarded_jwt: Option<(String, String)>,
}

impl OpenApiBridge {
    pub fn new(entry: Arc<ResourceEntry<McpServer>>) -> Self {
        let timeout = entry
            .value
            .timeout_ms
            .map(Duration::from_millis)
            .unwrap_or(crate::bridge::DEFAULT_UPSTREAM_TIMEOUT);
        Self {
            entry,
            timeout,
            forwarded_jwt: None,
        }
    }

    /// Deliver the calling agent's verified JWT, as already resolved to a
    /// header name and value by `aisix_core::forwarded_jwt`.
    pub fn with_forwarded_jwt(mut self, forwarded: Option<(String, String)>) -> Self {
        self.forwarded_jwt = forwarded;
        self
    }

    fn server(&self) -> &McpServer {
        &self.entry.value
    }

    fn tools(&self) -> Result<Vec<GeneratedTool>, McpError> {
        let spec = self.server().spec.as_ref().ok_or_else(|| {
            McpError::Request("openapi server has no spec configured".to_string())
        })?;
        generate_tools(spec)
    }

    /// Inject the gateway-held credential for this server. For `oauth2` this
    /// mints (or reuses) an access token via the shared token cache.
    async fn apply_auth(
        &self,
        request: reqwest::RequestBuilder,
    ) -> Result<reqwest::RequestBuilder, McpError> {
        let server = self.server();
        // The caller's JWT wins the slot it names: `RequestBuilder::header`
        // appends, so filling a slot twice would put two credentials on the
        // wire and let the REST API pick between them.
        let jwt_slot = self.forwarded_jwt.as_ref().map(|(name, _)| name.as_str());
        let api_key_header = server
            .api_key_header
            .as_deref()
            .filter(|h| !h.is_empty())
            .unwrap_or(DEFAULT_API_KEY_HEADER);
        let request = match server.auth_type {
            McpAuthType::None => request,
            McpAuthType::Bearer if jwt_slot == Some("authorization") => request,
            McpAuthType::Bearer => request.bearer_auth(server.secret.as_deref().unwrap_or("")),
            McpAuthType::ApiKey
                if jwt_slot == Some(api_key_header.to_ascii_lowercase().as_str()) =>
            {
                request
            }
            McpAuthType::ApiKey => {
                request.header(api_key_header, server.secret.as_deref().unwrap_or(""))
            }
            McpAuthType::OAuth2 if jwt_slot == Some("authorization") => request,
            McpAuthType::OAuth2 => {
                let token = crate::oauth::get_or_fetch(&self.oauth_config()).await?;
                request.bearer_auth(token)
            }
        };
        Ok(match &self.forwarded_jwt {
            // Built as a HeaderValue so `set_sensitive` survives: a `&str`
            // would have reqwest construct a fresh, unmarked value.
            Some((name, value)) => match reqwest::header::HeaderValue::from_str(value) {
                Ok(mut v) => {
                    v.set_sensitive(true);
                    request.header(name.as_str(), v)
                }
                Err(_) => request,
            },
            None => request,
        })
    }

    fn oauth_config(&self) -> OAuthClientConfig {
        let server = self.server();
        OAuthClientConfig {
            client_id: server.client_id.clone().unwrap_or_default(),
            client_secret: server.secret.clone().unwrap_or_default(),
            token_url: server.token_url.clone().unwrap_or_default(),
            scopes: server.scopes.clone().unwrap_or_default(),
        }
    }

    async fn execute(
        &self,
        tool: &GeneratedTool,
        arguments: &Value,
    ) -> Result<McpToolResult, McpError> {
        let args = match arguments {
            Value::Object(map) => map.clone(),
            Value::Null => Map::new(),
            _ => {
                return Err(McpError::Request(
                    "tool arguments must be a JSON object or null".to_string(),
                ))
            }
        };

        // An argument-shaped failure is a tool-level error result, not a
        // protocol error: the agent sees the message and can correct the
        // call, mirroring how a non-2xx response is surfaced.
        let url = match build_url(&self.server().url, tool, &args) {
            Ok(url) => url,
            Err(message) => return Ok(tool_error(message)),
        };
        let method = reqwest::Method::from_bytes(tool.method.to_uppercase().as_bytes())
            .map_err(|_| McpError::Request(format!("unsupported HTTP method {}", tool.method)))?;

        let mut request = http_client().request(method, url);
        request = self.apply_auth(request).await?;

        let query = build_query_pairs(tool, &args);
        if !query.is_empty() {
            request = request.query(&query);
        }

        if tool.has_body {
            if let Some(body) = coerce_body(args.get("body")) {
                request = request.json(&body);
            }
        }

        // The error text (which may embed the operator-configured base URL)
        // is logged server-side by the gateway and never returned to the
        // agent; sanitizing bounds it and strips log-injection vectors.
        let response = request.send().await.map_err(|e| {
            McpError::Request(format!(
                "HTTP request failed: {}",
                crate::bridge::sanitize_error_message(&e.to_string())
            ))
        })?;

        let status = response.status();
        // Mirrors the connect-time posture in `bridge.rs`: a 401 against a
        // minted token means it was revoked early — drop the cache entry so
        // the next call re-mints instead of replaying it.
        if status == reqwest::StatusCode::UNAUTHORIZED
            && self.server().auth_type == McpAuthType::OAuth2
        {
            crate::oauth::invalidate(&self.oauth_config());
        }
        let body = response.text().await.map_err(|e| {
            McpError::Request(format!(
                "failed to read response: {}",
                crate::bridge::sanitize_error_message(&e.to_string())
            ))
        })?;

        if status.is_success() {
            Ok(McpToolResult {
                content: json!([{ "type": "text", "text": body }]),
                structured_content: None,
                is_error: false,
            })
        } else {
            Ok(tool_error(format!("HTTP {}: {}", status.as_u16(), body)))
        }
    }
}

/// A tool-level error result (`isError: true` with a text message) — the
/// agent-visible failure shape for bad arguments and non-2xx responses.
fn tool_error(text: String) -> McpToolResult {
    McpToolResult {
        content: json!([{ "type": "text", "text": text }]),
        structured_content: None,
        is_error: true,
    }
}

#[async_trait]
impl McpBridge for OpenApiBridge {
    async fn list_tools(&self) -> Result<Vec<McpTool>, McpError> {
        Ok(self
            .tools()?
            .into_iter()
            .map(|t| McpTool {
                name: t.name,
                description: Some(t.description),
                input_schema: t.input_schema,
            })
            .collect())
    }

    async fn call_tool(&self, name: &str, arguments: Value) -> Result<McpToolResult, McpError> {
        let tools = self.tools()?;
        let tool = tools
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| McpError::Request(format!("unknown tool '{name}'")))?;
        tokio::time::timeout(self.timeout, self.execute(tool, &arguments))
            .await
            .map_err(|_| McpError::Request("tool call timed out".to_string()))?
    }
}

/// Map an `operationId` (or fallback) to a provider-safe tool name:
/// lowercase, any character outside `[a-zA-Z0-9_-]` replaced with `_`,
/// capped at [`TOOL_NAME_MAX_LEN`]. Mirrors LiteLLM's sanitizer.
fn sanitize_tool_name(raw: &str) -> String {
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

/// Strictly validate an OpenAPI document for `type: openapi` registration,
/// returning the generated tool names on success.
///
/// This is the write-path (Admin API / control plane) contract: a document
/// with no usable `paths`, zero generatable operations, or post-sanitization
/// tool-name collisions is rejected with a message naming the problem —
/// unlike [`generate_tools`], whose runtime posture is to degrade.
pub fn validate_spec(spec: &Value) -> Result<Vec<String>, McpError> {
    let generation = generate(spec)?;
    if !generation.duplicates.is_empty() {
        return Err(McpError::Request(format!(
            "duplicate tool names after operationId sanitization: {} — make the \
             operationIds distinct under lowercase [a-z0-9_-]",
            generation.duplicates.join(", ")
        )));
    }
    if generation.tools.is_empty() {
        let mut message =
            "spec has no operations that can become tools (methods get/post/put/delete/patch)"
                .to_string();
        if !generation.skipped.is_empty() {
            message.push_str(&format!(
                "; skipped operations without an application/json request body: {}",
                generation.skipped.join(", ")
            ));
        }
        return Err(McpError::Request(message));
    }
    Ok(generation.tools.into_iter().map(|t| t.name).collect())
}

/// Generate the tool set from an OpenAPI 3.x document.
///
/// Anomalies inside a single operation (a request body without an
/// `application/json` variant, an unresolvable parameter ref) skip or degrade
/// that operation only; the error path is reserved for a document that has no
/// usable `paths` object at all. Name collisions after sanitization are
/// disambiguated with `_2` / `_3` … suffixes — the write path rejects them
/// via [`validate_spec`], so this only defends rows written past it.
pub(crate) fn generate_tools(spec: &Value) -> Result<Vec<GeneratedTool>, McpError> {
    Ok(generate(spec)?.tools)
}

/// Outcome of walking a spec: the tools plus the anomalies the strict write
/// path reports (and the runtime path merely logs).
struct Generation {
    tools: Vec<GeneratedTool>,
    /// Base names that collided after sanitization (each listed once).
    duplicates: Vec<String>,
    /// `<METHOD> <path>` of operations skipped for an unsupported body.
    skipped: Vec<String>,
}

fn generate(spec: &Value) -> Result<Generation, McpError> {
    let paths = spec
        .get("paths")
        .and_then(Value::as_object)
        .ok_or_else(|| McpError::Request("openapi spec has no `paths` object".to_string()))?;

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
                        tracing::debug!(
                            path = %path,
                            method = %method,
                            "skipping operation: request body has no application/json content"
                        );
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

/// Substitute path parameters into the template and join with the base URL.
///
/// Every `{param}` in the template must be supplied: OpenAPI path parameters
/// are required by definition, and leaving a literal `{param}` in the URL
/// (LiteLLM's behavior) produces a request that can only 404. Values are
/// checked against traversal (`/`, `\`, `.`, `..`) and percent-encoded.
fn build_url(
    base_url: &str,
    tool: &GeneratedTool,
    args: &Map<String, Value>,
) -> Result<String, String> {
    let mut path = tool.path.clone();
    for param in &tool.path_params {
        let value = args.get(param.as_str()).unwrap_or(&Value::Null);
        let raw = match value {
            Value::String(s) => s.clone(),
            Value::Number(n) => n.to_string(),
            Value::Bool(b) => b.to_string(),
            Value::Null => return Err(format!("missing required path parameter '{param}'")),
            _ => {
                return Err(format!(
                    "path parameter '{param}' must be a string, number, or boolean"
                ))
            }
        };
        let safe = sanitize_path_value(&raw, param)?;
        path = path.replace(&format!("{{{param}}}"), &safe);
    }
    Ok(format!("{}{}", base_url.trim_end_matches('/'), path))
}

/// Reject path values that could change the request target (segment
/// separators, `.`/`..`), then percent-encode the rest.
fn sanitize_path_value(raw: &str, param: &str) -> Result<String, String> {
    if raw.is_empty() {
        return Err(format!("missing required path parameter '{param}'"));
    }
    if raw.contains('/') || raw.contains('\\') {
        return Err(format!(
            "path parameter '{param}' must not contain path separators"
        ));
    }
    if raw == "." || raw == ".." {
        return Err(format!("path parameter '{param}' cannot be '.' or '..'"));
    }
    Ok(utf8_percent_encode(raw, PATH_SEGMENT_ENCODE).to_string())
}

/// Build the query string pairs from the declared query parameters present in
/// the arguments: scalars serialize plainly, arrays repeat the key per item,
/// and objects are JSON-encoded.
fn build_query_pairs(tool: &GeneratedTool, args: &Map<String, Value>) -> Vec<(String, String)> {
    let mut pairs = Vec::new();
    for param in &tool.query_params {
        let Some(value) = args.get(param.as_str()) else {
            continue;
        };
        match value {
            Value::Null => {}
            Value::Array(items) => {
                for item in items {
                    pairs.push((param.clone(), scalar_to_string(item)));
                }
            }
            other => pairs.push((param.clone(), scalar_to_string(other))),
        }
    }
    pairs
}

fn scalar_to_string(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Coerce the `body` argument into the JSON body to send, mirroring LiteLLM:
/// objects and arrays pass through; a string is parsed as JSON when possible
/// and wrapped as `{"data": <string>}` otherwise; other scalars wrap the same
/// way; null/absent means no body.
fn coerce_body(value: Option<&Value>) -> Option<Value> {
    match value? {
        Value::Null => None,
        v @ (Value::Object(_) | Value::Array(_)) => Some(v.clone()),
        Value::String(s) => match serde_json::from_str::<Value>(s) {
            Ok(parsed) => Some(parsed),
            Err(_) => Some(json!({ "data": s })),
        },
        scalar => Some(json!({ "data": scalar })),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool_names(spec: &Value) -> Vec<String> {
        generate_tools(spec)
            .unwrap()
            .into_iter()
            .map(|t| t.name)
            .collect()
    }

    fn find<'a>(tools: &'a [GeneratedTool], name: &str) -> &'a GeneratedTool {
        tools
            .iter()
            .find(|t| t.name == name)
            .unwrap_or_else(|| panic!("tool {name} not generated"))
    }

    #[test]
    fn sanitizes_operation_ids_like_litellm() {
        // GitHub-style tag-namespaced ids gain `_` for `/`; uppercase folds.
        assert_eq!(
            sanitize_tool_name("actions/Download-Job.Logs"),
            "actions_download-job_logs"
        );
        let long = "x".repeat(200);
        assert_eq!(sanitize_tool_name(&long).len(), TOOL_NAME_MAX_LEN);
    }

    #[test]
    fn generates_tools_with_fallback_names_and_descriptions() {
        let spec = json!({
            "openapi": "3.0.0",
            "paths": {
                "/items": {
                    "get": { "operationId": "listItems", "summary": "List items" },
                    // No operationId: name falls back to `<method>_<path>`.
                    "post": {}
                }
            }
        });
        let tools = generate_tools(&spec).unwrap();
        let names = tools.iter().map(|t| t.name.as_str()).collect::<Vec<_>>();
        assert!(names.contains(&"listitems"), "{names:?}");
        assert!(names.contains(&"post__items"), "{names:?}");
        assert_eq!(find(&tools, "listitems").description, "List items");
        assert_eq!(find(&tools, "post__items").description, "POST /items");
    }

    #[test]
    fn disambiguates_sanitized_name_collisions() {
        let spec = json!({
            "paths": {
                "/a": { "get": { "operationId": "foo/list" } },
                "/b": { "get": { "operationId": "foo.list" } }
            }
        });
        let mut names = tool_names(&spec);
        names.sort();
        assert_eq!(names, vec!["foo_list", "foo_list_2"]);
    }

    #[test]
    fn builds_schema_from_params_and_body() {
        let spec = json!({
            "paths": {
                "/items/{id}": {
                    // Path-level param applies to the operation.
                    "parameters": [
                        { "name": "id", "in": "path", "required": true,
                          "schema": { "type": "integer" } }
                    ],
                    "patch": {
                        "operationId": "updateItem",
                        "parameters": [
                            { "name": "dry_run", "in": "query",
                              "description": "Validate only",
                              "schema": { "type": "boolean" } },
                            // Header params are the gateway's, not the agent's.
                            { "name": "x-tenant", "in": "header",
                              "schema": { "type": "string" } }
                        ],
                        "requestBody": {
                            "required": true,
                            "description": "Fields to update",
                            "content": { "application/json": {
                                "schema": { "type": "object",
                                    "properties": { "note": { "type": "string" } },
                                    "required": ["note"] } } }
                        }
                    }
                }
            }
        });
        let tools = generate_tools(&spec).unwrap();
        let tool = find(&tools, "updateitem");
        assert_eq!(tool.method, "patch");
        assert_eq!(tool.path, "/items/{id}");
        assert_eq!(tool.path_params, vec!["id"]);
        assert_eq!(tool.query_params, vec!["dry_run"]);
        assert!(tool.has_body);

        let schema = &tool.input_schema;
        let props = schema.get("properties").unwrap();
        assert_eq!(props["id"]["type"], "integer");
        assert_eq!(props["dry_run"]["type"], "boolean");
        assert_eq!(props["dry_run"]["description"], "Validate only");
        assert!(props.get("x-tenant").is_none(), "header params excluded");
        assert_eq!(props["body"]["type"], "object");
        assert_eq!(props["body"]["description"], "Fields to update");
        assert_eq!(props["body"]["properties"]["note"]["type"], "string");
        let required = schema.get("required").unwrap().as_array().unwrap();
        assert!(required.contains(&json!("id")));
        assert!(required.contains(&json!("body")));
        assert!(!required.contains(&json!("dry_run")));
    }

    #[test]
    fn operation_params_override_path_level_on_same_name() {
        let spec = json!({
            "paths": {
                "/x/{v}": {
                    "parameters": [
                        { "name": "v", "in": "path", "required": true,
                          "schema": { "type": "string" } }
                    ],
                    "get": {
                        "operationId": "getX",
                        "parameters": [
                            { "name": "v", "in": "path", "required": true,
                              "schema": { "type": "integer" } }
                        ]
                    }
                }
            }
        });
        let tools = generate_tools(&spec).unwrap();
        let tool = find(&tools, "getx");
        assert_eq!(tool.input_schema["properties"]["v"]["type"], "integer");
        assert_eq!(tool.path_params, vec!["v"]);
    }

    #[test]
    fn resolves_component_refs_in_params_and_body() {
        let spec = json!({
            "components": {
                "parameters": {
                    "PerPage": { "name": "per_page", "in": "query",
                                 "schema": { "type": "integer" } }
                },
                "schemas": {
                    "Order": { "type": "object",
                        "properties": {
                            "sku": { "type": "string" },
                            "customer": { "$ref": "#/components/schemas/Customer" }
                        } },
                    "Customer": { "type": "object",
                        "properties": { "name": { "type": "string" } } }
                }
            },
            "paths": {
                "/orders": {
                    "post": {
                        "operationId": "createOrder",
                        "parameters": [ { "$ref": "#/components/parameters/PerPage" } ],
                        "requestBody": { "content": { "application/json": {
                            "schema": { "$ref": "#/components/schemas/Order" } } } }
                    }
                }
            }
        });
        let tools = generate_tools(&spec).unwrap();
        let tool = find(&tools, "createorder");
        assert_eq!(tool.query_params, vec!["per_page"]);
        let body = &tool.input_schema["properties"]["body"];
        assert_eq!(body["properties"]["sku"]["type"], "string");
        // Nested ref resolved one level deeper.
        assert_eq!(
            body["properties"]["customer"]["properties"]["name"]["type"],
            "string"
        );
    }

    #[test]
    fn cyclic_refs_degrade_to_empty_schema_instead_of_hanging() {
        let spec = json!({
            "components": { "schemas": {
                "Node": { "type": "object",
                    "properties": { "next": { "$ref": "#/components/schemas/Node" } } }
            } },
            "paths": { "/nodes": { "post": {
                "operationId": "createNode",
                "requestBody": { "content": { "application/json": {
                    "schema": { "$ref": "#/components/schemas/Node" } } } }
            } } }
        });
        let tools = generate_tools(&spec).unwrap();
        // Terminates; the innermost expansion bottoms out at `{}`.
        let body = &tools[0].input_schema["properties"]["body"];
        assert_eq!(body["type"], "object");
    }

    #[test]
    fn unresolvable_and_external_refs_degrade_to_any() {
        let spec = json!({
            "paths": { "/a": { "post": {
                "operationId": "a",
                "requestBody": { "content": { "application/json": {
                    "schema": { "$ref": "https://elsewhere.example/schema.json" } } } }
            } } }
        });
        let tools = generate_tools(&spec).unwrap();
        // External ref → `{}` → coerced to an object schema for `body`.
        assert_eq!(
            tools[0].input_schema["properties"]["body"]["type"],
            "object"
        );
    }

    #[test]
    fn skips_operations_without_a_json_body_variant() {
        let spec = json!({
            "paths": {
                "/upload": { "post": {
                    "operationId": "uploadFile",
                    "requestBody": { "content": { "multipart/form-data": {
                        "schema": { "type": "object" } } } }
                } },
                "/ok": { "post": {
                    "operationId": "jsonVariant",
                    "requestBody": { "content": {
                        "application/json; charset=utf-8": { "schema": { "type": "object" } }
                    } }
                } },
                "/problem": { "post": {
                    "operationId": "problemJson",
                    "requestBody": { "content": {
                        "application/problem+json": { "schema": { "type": "object" } }
                    } }
                } }
            }
        });
        let mut names = tool_names(&spec);
        names.sort();
        // `uploadFile` is absent; parameterized and `+json` variants count.
        assert_eq!(names, vec!["jsonvariant", "problemjson"]);
    }

    #[test]
    fn spec_without_paths_is_an_error() {
        let err = generate_tools(&json!({ "openapi": "3.0.0" })).unwrap_err();
        assert!(err.to_string().contains("paths"), "{err}");
    }

    #[test]
    fn build_url_substitutes_encodes_and_rejects_traversal() {
        let tool = GeneratedTool {
            name: "t".into(),
            description: String::new(),
            input_schema: json!({}),
            method: "get".into(),
            path: "/items/{id}/sub".into(),
            path_params: vec!["id".into()],
            query_params: vec![],
            has_body: false,
        };
        let args = |v: Value| {
            let mut m = Map::new();
            m.insert("id".into(), v);
            m
        };

        assert_eq!(
            build_url("https://api.example.com/v1/", &tool, &args(json!("a b#c"))).unwrap(),
            "https://api.example.com/v1/items/a%20b%23c/sub"
        );
        assert_eq!(
            build_url("https://api.example.com", &tool, &args(json!(42))).unwrap(),
            "https://api.example.com/items/42/sub"
        );
        for bad in [
            json!("../etc"),
            json!("a/b"),
            json!("a\\b"),
            json!(".."),
            json!("."),
        ] {
            assert!(
                build_url("https://api.example.com", &tool, &args(bad.clone())).is_err(),
                "expected rejection for {bad}"
            );
        }
        let missing = Map::new();
        let err = build_url("https://api.example.com", &tool, &missing).unwrap_err();
        assert!(err.to_string().contains("missing required path parameter"));
    }

    #[test]
    fn query_pairs_serialize_scalars_arrays_and_objects() {
        let tool = GeneratedTool {
            name: "t".into(),
            description: String::new(),
            input_schema: json!({}),
            method: "get".into(),
            path: "/".into(),
            path_params: vec![],
            query_params: vec!["q".into(), "tags".into(), "filter".into(), "absent".into()],
            has_body: false,
        };
        let mut args = Map::new();
        args.insert("q".into(), json!("text"));
        args.insert("tags".into(), json!(["a", 2]));
        args.insert("filter".into(), json!({"k": "v"}));
        args.insert("undeclared".into(), json!("dropped"));
        assert_eq!(
            build_query_pairs(&tool, &args),
            vec![
                ("q".to_string(), "text".to_string()),
                ("tags".to_string(), "a".to_string()),
                ("tags".to_string(), "2".to_string()),
                ("filter".to_string(), r#"{"k":"v"}"#.to_string()),
            ]
        );
    }

    #[test]
    fn validate_spec_rejects_duplicates_and_empty_specs() {
        // Collision: strict validation names the colliding base name.
        let dup = json!({
            "paths": {
                "/a": { "get": { "operationId": "foo/list" } },
                "/b": { "get": { "operationId": "foo.list" } }
            }
        });
        let err = validate_spec(&dup).unwrap_err().to_string();
        assert!(err.contains("duplicate tool names"), "{err}");
        assert!(err.contains("foo_list"), "{err}");

        // No paths at all.
        assert!(validate_spec(&json!({ "openapi": "3.0.0" })).is_err());

        // Paths but nothing generatable — the skipped multipart op is named.
        let only_multipart = json!({
            "paths": { "/upload": { "post": {
                "operationId": "up",
                "requestBody": { "content": { "multipart/form-data": {} } }
            } } }
        });
        let err = validate_spec(&only_multipart).unwrap_err().to_string();
        assert!(err.contains("no operations"), "{err}");
        assert!(err.contains("POST /upload"), "{err}");

        // Healthy spec: returns the generated names.
        let ok = json!({
            "paths": { "/items": { "get": { "operationId": "listItems" } } }
        });
        assert_eq!(validate_spec(&ok).unwrap(), vec!["listitems"]);
    }

    #[test]
    fn coerce_body_matches_litellm_semantics() {
        assert_eq!(coerce_body(Some(&json!({"a": 1}))), Some(json!({"a": 1})));
        assert_eq!(
            coerce_body(Some(&json!(r#"{"parsed": true}"#))),
            Some(json!({"parsed": true}))
        );
        assert_eq!(
            coerce_body(Some(&json!("plain text"))),
            Some(json!({"data": "plain text"}))
        );
        assert_eq!(coerce_body(Some(&json!(5))), Some(json!({"data": 5})));
        assert_eq!(coerce_body(Some(&Value::Null)), None);
        assert_eq!(coerce_body(None), None);
    }
}
