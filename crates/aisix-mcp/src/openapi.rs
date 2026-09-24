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
//! the tool set only changes when the resource does. The generation itself is
//! [`aisix_core::mcp_openapi`], shared with the resource loaders, which reject
//! a document it yields no tools from.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use aisix_core::mcp_openapi::{generate, GeneratedTool};
use aisix_core::snapshot::ResourceTable;
use aisix_core::{McpAuthType, McpServer, ResourceEntry};
use async_trait::async_trait;
use percent_encoding::{utf8_percent_encode, AsciiSet, CONTROLS};
use serde_json::{json, Map, Value};

use crate::bridge::{McpBridge, McpTool, McpToolResult, OAuthClientConfig};
use crate::error::McpError;

/// Header the API key is sent under for `auth_type: api_key` when the
/// resource sets no `api_key_header`.
pub const DEFAULT_API_KEY_HEADER: &str = "x-api-key";

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

/// [`McpBridge`] over an OpenAPI-backed `mcp_server` resource. Holds the
/// snapshot entry (`Arc`, shared with the snapshot) so the spec is never
/// deep-cloned per request.
pub struct OpenApiBridge {
    entry: Arc<ResourceEntry<McpServer>>,
    timeout: Duration,
    /// Inbound client headers this server's `forward_client_headers`
    /// admits, resolved per request against the calling agent's own
    /// request and delivered to the REST API behind these tools.
    forwarded_client_headers: Vec<(reqwest::header::HeaderName, reqwest::header::HeaderValue)>,
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
            forwarded_client_headers: Vec::new(),
        }
    }

    /// Deliver the client headers this server forwards, as already
    /// resolved against the inbound request.
    pub fn with_forwarded_client_headers(
        mut self,
        forwarded: Vec<(reqwest::header::HeaderName, reqwest::header::HeaderValue)>,
    ) -> Self {
        self.forwarded_client_headers = forwarded;
        self
    }

    fn server(&self) -> &McpServer {
        &self.entry.value
    }

    /// This server's tool set, generated once per row version.
    ///
    /// The aggregating `/mcp` endpoint builds a bridge per enabled server
    /// per request, and both `tools/list` and `tools/call` ask for the
    /// tools — so without this every call walked every registered
    /// server's whole OpenAPI document, resolving `$ref`s and building a
    /// JSON Schema per operation (AISIX-Cloud#1542).
    fn tools(&self) -> Result<Arc<Vec<GeneratedTool>>, McpError> {
        let spec = self.server().spec.as_ref().ok_or_else(|| {
            McpError::Request("openapi server has no spec configured".to_string())
        })?;
        // Row identity is the snapshot's `Arc`: a copy-on-write publish
        // shares rows it did not change, so the same allocation means the
        // same spec. The cache keeps its own `Arc`, so the address cannot
        // be recycled underneath it.
        if let Some(hit) = lookup_tools(&self.entry) {
            return Ok(hit);
        }
        // Generated with the lock RELEASED. One lock serves every
        // registered server, so holding it across the spec walk would
        // make one server's miss block every other server's hit — and a
        // write that replaces one row makes the aggregating endpoint's
        // concurrent requests all miss at once. The cost is that a race
        // may generate the same tool set twice; the second insert simply
        // replaces the first, and both are equal.
        let tools = Arc::new(generate_tools(spec)?);
        let mut cache = tool_cache();
        cache.by_id.insert(
            self.entry.id.clone(),
            CachedTools {
                row: Arc::clone(&self.entry),
                tools: Arc::clone(&tools),
            },
        );
        Ok(tools)
    }

    /// Inject the gateway-held credential for this server. For `oauth2` this
    /// mints (or reuses) an access token via the shared token cache.
    async fn apply_auth(
        &self,
        request: reqwest::RequestBuilder,
    ) -> Result<reqwest::RequestBuilder, McpError> {
        let server = self.server();
        // A forwarded client header wins the slot it names:
        // `RequestBuilder::header` appends, so filling a slot twice would
        // put two credentials on the wire and let the REST API pick
        // between them.
        //
        // The claimed set is read before anything fills a slot, so
        // suppressing the server credential and delivering the forwarded
        // value cannot disagree.
        let forwarded = &self.forwarded_client_headers;
        let claims = |name: &str| forwarded.iter().any(|(n, _)| n.as_str() == name);
        let api_key_header = server
            .api_key_header
            .as_deref()
            .filter(|h| !h.is_empty())
            .unwrap_or(DEFAULT_API_KEY_HEADER);
        let request = match server.auth_type {
            McpAuthType::None => request,
            McpAuthType::Bearer if claims("authorization") => request,
            McpAuthType::Bearer => request.bearer_auth(server.secret.as_deref().unwrap_or("")),
            McpAuthType::ApiKey if claims(api_key_header.to_ascii_lowercase().as_str()) => request,
            McpAuthType::ApiKey => {
                request.header(api_key_header, server.secret.as_deref().unwrap_or(""))
            }
            McpAuthType::OAuth2 if claims("authorization") => request,
            McpAuthType::OAuth2 => {
                let token = crate::oauth::get_or_fetch(&self.oauth_config()).await?;
                request.bearer_auth(token)
            }
        };
        Ok(forwarded.iter().fold(request, |req, (name, value)| {
            req.header(name.clone(), value.clone())
        }))
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

/// Tool sets generated from `type: openapi` rows, keyed by the row's
/// etcd id and shared across every per-request bridge in the process.
static TOOL_CACHE: OnceLock<Mutex<ToolCache>> = OnceLock::new();

#[derive(Default)]
struct ToolCache {
    /// The `mcp_servers` table generation the map was last swept against.
    swept: Option<u64>,
    by_id: HashMap<String, CachedTools>,
}

struct CachedTools {
    row: Arc<ResourceEntry<McpServer>>,
    tools: Arc<Vec<GeneratedTool>>,
}

/// The cached tool set for `entry`, if this exact row version has one.
/// Holds the lock only long enough to read it.
fn lookup_tools(entry: &Arc<ResourceEntry<McpServer>>) -> Option<Arc<Vec<GeneratedTool>>> {
    let cache = tool_cache();
    let cached = cache.by_id.get(&entry.id)?;
    Arc::ptr_eq(&cached.row, entry).then(|| Arc::clone(&cached.tools))
}

fn tool_cache() -> std::sync::MutexGuard<'static, ToolCache> {
    TOOL_CACHE
        .get_or_init(|| Mutex::new(ToolCache::default()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Forget cached tool sets for servers the snapshot no longer carries.
///
/// Called where a snapshot is in hand — the bridge itself only ever sees
/// its own row. Cheap when nothing changed: a create/delete moves the
/// `mcp_servers` generation and nothing else does.
pub(crate) fn sweep_tool_cache(servers: &ResourceTable<McpServer>) {
    let generation = servers.generation();
    let mut cache = tool_cache();
    if cache.swept == Some(generation) {
        return;
    }
    cache.by_id.retain(|id, _| servers.get_by_id(id).is_some());
    cache.swept = Some(generation);
}

#[async_trait]
impl McpBridge for OpenApiBridge {
    async fn list_tools(&self) -> Result<Vec<McpTool>, McpError> {
        Ok(self
            .tools()?
            .iter()
            .map(|t| McpTool {
                name: t.name.clone(),
                description: Some(t.description.clone()),
                input_schema: t.input_schema.clone(),
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

/// Generate the tool set from an OpenAPI 3.x document — the same walk the
/// loaders judge a spec by ([`aisix_core::mcp_openapi::generate`]), so a
/// row that loaded always yields at least one tool here. Name collisions
/// after sanitization are disambiguated with `_2` / `_3` … suffixes and
/// every tool stays callable.
pub(crate) fn generate_tools(spec: &Value) -> Result<Vec<GeneratedTool>, McpError> {
    let generation = generate(spec).map_err(McpError::Request)?;
    for skipped in &generation.skipped {
        tracing::debug!(
            operation = %skipped,
            "skipping operation: request body has no application/json content"
        );
    }
    Ok(generation.tools)
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

    fn openapi_server(id: &str, path: &str) -> Arc<ResourceEntry<McpServer>> {
        let server: McpServer = serde_json::from_value(json!({
            "name": format!("srv-{id}"),
            "type": "openapi",
            "url": "http://127.0.0.1:1/",
            "spec": {
                "openapi": "3.0.0",
                "paths": { path: { "get": { "operationId": "listItems" } } }
            }
        }))
        .expect("test mcp_server");
        Arc::new(ResourceEntry::new(id, server, 1))
    }

    fn server_table(entries: &[Arc<ResourceEntry<McpServer>>]) -> ResourceTable<McpServer> {
        let table = ResourceTable::new();
        for e in entries {
            table.insert_arc(Arc::clone(e));
        }
        table
    }

    /// The aggregating `/mcp` endpoint builds a bridge per enabled server
    /// per REQUEST, and both `tools/list` and `tools/call` ask for the
    /// tool set — so regenerating it from the spec each time made one
    /// call cost a walk of every registered document (AISIX-Cloud#1542).
    // Uses ids of its own: `TOOL_CACHE` is process-wide, and
    // `sweep_tool_cache` with a table that does not carry a row evicts it.
    // A second test in THIS crate's lib-test binary that reaches the cache
    // (anything going through `McpGateway::from_snapshot*`) would have to
    // coordinate with this one; today there is none.
    #[test]
    fn tool_generation_is_cached_per_row_and_evicted_with_it() {
        let row = openapi_server("cache-row-1", "/items");
        let table = server_table(&[Arc::clone(&row)]);
        sweep_tool_cache(&table);

        // Two bridges, as two requests would build them.
        let first = OpenApiBridge::new(Arc::clone(&row)).tools().unwrap();
        let second = OpenApiBridge::new(Arc::clone(&row)).tools().unwrap();
        assert!(Arc::ptr_eq(&first, &second), "the tool set was regenerated");
        assert_eq!(first[0].name, "listitems");

        // A rewritten row is a different `Arc`: regenerate.
        let rewritten = openapi_server("cache-row-1", "/things");
        let after_write = OpenApiBridge::new(Arc::clone(&rewritten)).tools().unwrap();
        assert!(!Arc::ptr_eq(&first, &after_write));
        assert_eq!(after_write[0].path, "/things");

        // Deleting the row evicts it, so the map cannot grow under
        // create/delete churn.
        sweep_tool_cache(&server_table(&[]));
        let after_delete = OpenApiBridge::new(Arc::clone(&rewritten)).tools().unwrap();
        assert!(
            !Arc::ptr_eq(&after_write, &after_delete),
            "the deleted row was still cached",
        );
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
