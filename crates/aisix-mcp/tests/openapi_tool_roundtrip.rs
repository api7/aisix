//! End-to-end test of an OpenAPI-backed server behind the MCP gateway: a real
//! REST API (axum, ephemeral port) is registered as a `type: openapi`
//! `mcp_server`, and a real rmcp client drives the gateway's `/mcp` endpoint.
//!
//! Pins the issue-level acceptance criteria at the crate boundary:
//! - `tools/list` exposes one namespaced tool per spec operation with the
//!   generated input schema;
//! - `tools/call` executes the REST request — path substitution, query
//!   parameters, JSON body, and the gateway-held credential (never supplied
//!   by the agent);
//! - a non-2xx response and an argument mistake surface as tool-level errors
//!   (`isError: true`), not protocol errors.

use std::collections::HashMap;
use std::net::SocketAddr;

use aisix_core::{AisixSnapshot, McpServer, ResourceEntry};
use aisix_mcp::{streamable_http_service, McpGateway};
use axum::extract::{Path, Query};
use axum::http::{HeaderMap, StatusCode};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::Json;
use rmcp::model::{CallToolRequestParams, CallToolResult};
use rmcp::transport::StreamableHttpClientTransport;
use rmcp::ServiceExt;
use serde_json::{json, Value};

/// The bearer token the gateway holds for the fake ERP API. The REST handlers
/// 401 without it, proving the credential is injected gateway-side.
const ERP_TOKEN: &str = "tok-erp-123";

/// The custom API-key header (and key) for the second, `api_key`-mode server.
const INVENTORY_HEADER: &str = "x-inventory-key";
const INVENTORY_KEY: &str = "inv-key-456";

async fn serve(app: axum::Router) -> SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local addr");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("serve");
    });
    addr
}

/// A fake ERP REST API: bearer-authenticated echo endpoints.
async fn spawn_erp_api() -> SocketAddr {
    fn authed(headers: &HeaderMap) -> bool {
        headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .is_some_and(|v| v == format!("Bearer {ERP_TOKEN}"))
    }

    let app = axum::Router::new()
        .route(
            "/v1/items/:id",
            get(
                |headers: HeaderMap,
                 Path(id): Path<String>,
                 Query(q): Query<HashMap<String, String>>| async move {
                    if !authed(&headers) {
                        return (StatusCode::UNAUTHORIZED, Json(json!({"error": "no auth"})))
                            .into_response();
                    }
                    Json(json!({ "id": id, "query": q })).into_response()
                },
            ),
        )
        .route(
            "/v1/orders",
            post(|headers: HeaderMap, Json(body): Json<Value>| async move {
                if !authed(&headers) {
                    return (StatusCode::UNAUTHORIZED, Json(json!({"error": "no auth"})))
                        .into_response();
                }
                Json(json!({ "created": body })).into_response()
            }),
        )
        .route(
            "/v1/fail",
            get(|| async { (StatusCode::INTERNAL_SERVER_ERROR, "boom") }),
        );
    serve(app).await
}

/// The ERP API's OpenAPI document, as the control plane would materialize it.
fn erp_spec() -> Value {
    json!({
        "openapi": "3.0.0",
        "info": { "title": "ERP", "version": "1.0.0" },
        "paths": {
            "/items/{id}": {
                "get": {
                    "operationId": "getItem",
                    "summary": "Fetch one item",
                    "parameters": [
                        { "name": "id", "in": "path", "required": true,
                          "schema": { "type": "integer" } },
                        { "name": "verbose", "in": "query",
                          "schema": { "type": "boolean" } }
                    ]
                }
            },
            "/orders": {
                "post": {
                    "operationId": "createOrder",
                    "requestBody": {
                        "required": true,
                        "content": { "application/json": { "schema": {
                            "type": "object",
                            "properties": { "note": { "type": "string" } },
                            "required": ["note"]
                        } } }
                    }
                }
            },
            "/fail": { "get": { "operationId": "failOp" } }
        }
    })
}

fn openapi_entry(id: &str, config: Value) -> ResourceEntry<McpServer> {
    let server: McpServer = serde_json::from_value(config).expect("valid mcp_server resource");
    ResourceEntry::new(id, server, 1)
}

async fn spawn_gateway(gateway: McpGateway) -> SocketAddr {
    serve(axum::Router::new().nest_service("/mcp", streamable_http_service(gateway))).await
}

fn first_text(result: &CallToolResult) -> String {
    let value = serde_json::to_value(&result.content).expect("encode content");
    value[0]["text"].as_str().unwrap_or_default().to_string()
}

fn call(name: &str, args: Value) -> CallToolRequestParams {
    let mut params = CallToolRequestParams::new(name.to_string());
    if let Value::Object(map) = args {
        params = params.with_arguments(map);
    }
    params
}

#[tokio::test]
async fn openapi_server_lists_and_calls_generated_tools() {
    let api = spawn_erp_api().await;

    let snapshot = AisixSnapshot::new();
    snapshot.mcp_servers.insert(openapi_entry(
        "e1",
        json!({
            "name": "erp",
            "type": "openapi",
            "url": format!("http://{api}/v1"),
            "spec": erp_spec(),
            "auth_type": "bearer",
            "secret": ERP_TOKEN,
        }),
    ));

    let gw = spawn_gateway(McpGateway::from_snapshot(&snapshot)).await;
    let client = ()
        .serve(StreamableHttpClientTransport::from_uri(format!(
            "http://{gw}/mcp"
        )))
        .await
        .expect("connect downstream client");

    // tools/list: one namespaced tool per operation, schema preserved.
    let tools = client.list_tools(None).await.expect("list tools");
    let mut names: Vec<_> = tools.tools.iter().map(|t| t.name.to_string()).collect();
    names.sort();
    assert_eq!(
        names,
        vec!["erp__createorder", "erp__failop", "erp__getitem"]
    );

    let get_item = tools
        .tools
        .iter()
        .find(|t| t.name == "erp__getitem")
        .expect("getItem generated");
    let schema = serde_json::to_value(&get_item.input_schema).expect("schema");
    assert_eq!(schema["properties"]["id"]["type"], "integer");
    assert_eq!(schema["properties"]["verbose"]["type"], "boolean");
    assert_eq!(schema["required"], json!(["id"]));

    // tools/call GET: path substitution + query serialization + gateway-held
    // bearer (the client never sent a credential).
    let result = client
        .call_tool(call("erp__getitem", json!({ "id": 42, "verbose": true })))
        .await
        .expect("call getItem");
    assert_ne!(result.is_error, Some(true), "unexpected tool error");
    let echoed: Value = serde_json::from_str(&first_text(&result)).expect("json echo");
    assert_eq!(echoed["id"], "42");
    assert_eq!(echoed["query"]["verbose"], "true");

    // tools/call POST: the `body` argument becomes the JSON request body.
    let result = client
        .call_tool(call(
            "erp__createorder",
            json!({ "body": { "note": "hello" } }),
        ))
        .await
        .expect("call createOrder");
    assert_ne!(result.is_error, Some(true));
    let echoed: Value = serde_json::from_str(&first_text(&result)).expect("json echo");
    assert_eq!(echoed["created"]["note"], "hello");

    // Non-2xx → tool-level error carrying the status and body.
    let result = client
        .call_tool(call("erp__failop", Value::Null))
        .await
        .expect("call failOp");
    assert_eq!(result.is_error, Some(true));
    let text = first_text(&result);
    assert!(text.starts_with("HTTP 500:"), "got: {text}");
    assert!(text.contains("boom"), "got: {text}");

    // Argument mistake (missing required path param) → tool-level error the
    // agent can read and fix, not an opaque protocol error.
    let result = client
        .call_tool(call("erp__getitem", json!({ "verbose": true })))
        .await
        .expect("call getItem without id");
    assert_eq!(result.is_error, Some(true));
    assert!(
        first_text(&result).contains("missing required path parameter 'id'"),
        "got: {}",
        first_text(&result)
    );

    client.cancel().await.ok();
}

#[tokio::test]
async fn openapi_server_sends_api_key_under_configured_header() {
    // Echo back the configured custom header so the assertion sees exactly
    // what the gateway sent.
    let app = axum::Router::new().route(
        "/lookup",
        get(|headers: HeaderMap| async move {
            let key = headers
                .get(INVENTORY_HEADER)
                .and_then(|v| v.to_str().ok())
                .unwrap_or("")
                .to_string();
            Json(json!({ "received_key": key }))
        }),
    );
    let api = serve(app).await;

    let snapshot = AisixSnapshot::new();
    snapshot.mcp_servers.insert(openapi_entry(
        "e2",
        json!({
            "name": "inventory",
            "type": "openapi",
            "url": format!("http://{api}"),
            "spec": {
                "openapi": "3.0.0",
                "paths": { "/lookup": { "get": { "operationId": "lookup" } } }
            },
            "auth_type": "api_key",
            "secret": INVENTORY_KEY,
            "api_key_header": INVENTORY_HEADER,
        }),
    ));

    let gw = spawn_gateway(McpGateway::from_snapshot(&snapshot)).await;
    let client = ()
        .serve(StreamableHttpClientTransport::from_uri(format!(
            "http://{gw}/mcp"
        )))
        .await
        .expect("connect downstream client");

    let result = client
        .call_tool(call("inventory__lookup", Value::Null))
        .await
        .expect("call lookup");
    assert_ne!(result.is_error, Some(true));
    let echoed: Value = serde_json::from_str(&first_text(&result)).expect("json echo");
    assert_eq!(echoed["received_key"], INVENTORY_KEY);

    client.cancel().await.ok();
}

#[tokio::test]
async fn broken_spec_degrades_gracefully_next_to_healthy_servers() {
    // One healthy openapi server and one whose spec is unusable: the broken
    // one's tools are absent, the healthy one keeps serving — mirroring how a
    // dead real upstream degrades.
    let api = spawn_erp_api().await;

    let snapshot = AisixSnapshot::new();
    snapshot.mcp_servers.insert(openapi_entry(
        "e1",
        json!({
            "name": "erp",
            "type": "openapi",
            "url": format!("http://{api}/v1"),
            "spec": erp_spec(),
            "auth_type": "bearer",
            "secret": ERP_TOKEN,
        }),
    ));
    snapshot.mcp_servers.insert(openapi_entry(
        "e2",
        json!({
            "name": "broken",
            "type": "openapi",
            "url": "http://127.0.0.1:1/api",
            "spec": { "openapi": "3.0.0" }
        }),
    ));

    let gw = spawn_gateway(McpGateway::from_snapshot(&snapshot)).await;
    let client = ()
        .serve(StreamableHttpClientTransport::from_uri(format!(
            "http://{gw}/mcp"
        )))
        .await
        .expect("connect downstream client");

    let tools = client.list_tools(None).await.expect("list tools");
    let names: Vec<_> = tools.tools.iter().map(|t| t.name.to_string()).collect();
    assert!(names.iter().all(|n| n.starts_with("erp__")), "{names:?}");
    assert_eq!(names.len(), 3);

    client.cancel().await.ok();
}
