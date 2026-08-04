//! `/mcp` and `/mcp/{server}` — the downstream-facing MCP gateway endpoints.
//!
//! AISIX presents as a single MCP server to a downstream agent: it aggregates
//! the tools of the registered `mcp_servers` and routes tool calls back to
//! them. `/mcp/{server}` scopes the same gateway to one registered server,
//! serving its tools under their original (un-namespaced) names. The caller authenticates with an AISIX API key — the
//! [`AuthenticatedKey`] extractor rejects a missing or invalid key with `401`
//! before the request reaches the gateway. The gateway is rebuilt from the
//! current configuration snapshot on each request, so it always reflects the
//! live `mcp_servers` set.
//!
//! A `tools/call` is governed by the SAME pipeline as an LLM request, keyed on
//! the caller's API key: per-tool access control (the key's `allowed_tools`),
//! rate-limit + budget (`quota::enforce_mcp`, which adds the key's per-MCP-server
//! limit to the shared layers), guardrails on both the tool arguments (input)
//! and the tool result (output), and a usage event into the shared sink.

use std::time::{Duration, Instant};

use aisix_obs::{AccessLog, RequestOutcome, UsageEvent};
use axum::body::{to_bytes, Body};
use axum::extract::{Request, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use tower::ServiceExt;

use crate::auth::AuthenticatedKey;
use crate::request_id::new_request_id;
use crate::state::ProxyState;

/// Bounded `model` metric label for /mcp requests — MCP has no resolved
/// model, and the tool name is caller-controlled (unbounded Prometheus
/// cardinality, same rule as passthrough's #451 sentinel).
const MCP_MODEL_LABEL: &str = "mcp";

/// Just enough of a JSON-RPC request to tell a tool call apart from the MCP
/// handshake / discovery methods, recover the called tool's name + arguments,
/// and echo the request id back in a synthesized error. Unknown fields ignored.
#[derive(Deserialize)]
struct JsonRpcPeek {
    method: Option<String>,
    params: Option<PeekParams>,
    /// JSON-RPC request id, echoed back if the gateway synthesizes an error.
    id: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct PeekParams {
    /// The namespaced `<server>__<tool>` name on a `tools/call`.
    name: Option<String>,
    /// The tool arguments, scanned by input guardrails.
    arguments: Option<serde_json::Value>,
}

/// Serve a `/mcp` request. The [`AuthenticatedKey`] extractor enforces a valid
/// AISIX API key (responding `401` otherwise). A `tools/call` is then subject to
/// the same rate-limit and budget governance as an LLM request — keyed on the
/// caller's API key — before being handled by an MCP gateway built from the
/// current snapshot's `mcp_servers`, and a usage event is emitted into the same
/// pipeline as LLM calls. The `initialize` / `tools/list` handshake and discovery
/// methods pass through ungated and unmetered.
pub async fn mcp_endpoint(
    auth: AuthenticatedKey,
    State(state): State<ProxyState>,
    request: Request,
) -> Response {
    serve(auth, state, request, None).await
}

/// Serve a `/mcp/{server}` request: the single-server variant of
/// [`mcp_endpoint`]. The path names a registered MCP server; the gateway is
/// scoped to it and serves its tools under their original names (`tools/call`
/// also accepts the namespaced form). Everything else — auth, per-tool ACL,
/// quota, guardrails, usage — is the same pipeline as the aggregated endpoint;
/// only the server selection and the tool-name surface differ. An unknown or
/// disabled server is `404` (after auth, so an unauthenticated caller learns
/// nothing about which servers exist).
pub async fn mcp_scoped_endpoint(
    auth: AuthenticatedKey,
    axum::extract::Path(server): axum::extract::Path<String>,
    State(state): State<ProxyState>,
    request: Request,
) -> Response {
    serve(auth, state, request, Some(server)).await
}

async fn serve(
    auth: AuthenticatedKey,
    state: ProxyState,
    request: Request,
    scope: Option<String>,
) -> Response {
    // #698: /mcp emits the same access log + request metrics as every other
    // handler — pre-fix the endpoint was invisible in both. One wrapper
    // around `dispatch` covers every early-return path (quota, guardrail
    // blocks, gateway errors) with the actual response status.
    let started = Instant::now();
    let request_id = request
        .extensions()
        .get::<crate::request_id::RequestId>()
        .map(|r| r.0.clone())
        .unwrap_or_else(new_request_id);
    let api_key_id = auth.entry.id.clone();
    let method = request.method().clone();

    let response = dispatch(auth, scope.as_deref(), &state, request, &request_id).await;

    let elapsed = started.elapsed();
    let status = response.status().as_u16();
    AccessLog {
        method: method.as_str(),
        // Bounded route template, mirroring `/a2a` (the per-request server is
        // on the usage event, not the access log).
        path: if scope.is_some() {
            "/mcp/{server}"
        } else {
            "/mcp"
        },
        status,
        latency: elapsed,
        provider: Some("mcp"),
        model: None,
        api_key_id: Some(&api_key_id),
        prompt_tokens: None,
        completion_tokens: None,
        total_tokens: None,
        request_id: &request_id,
        // `dispatch` renders its own `Response` rather than surfacing a
        // `ProxyError`, so there is no typed error to name here. The status
        // code is all this endpoint can attribute a failure to.
        error_kind: None,
        error: None,
        served_by_model: None,
        routing_attempt_count: None,
        routing_fallback_count: None,
    }
    .emit();
    state.metrics.record_request(
        "mcp",
        MCP_MODEL_LABEL,
        status,
        RequestOutcome::from_status(status),
        elapsed,
    );
    response
}

async fn dispatch(
    auth: AuthenticatedKey,
    scope: Option<&str>,
    state: &ProxyState,
    request: Request,
    request_id: &str,
) -> Response {
    // One snapshot for the whole request: the scoped-server resolution below
    // and the gateway construction further down must see the same resource
    // set.
    let snapshot = state.snapshot.load();

    // Scoped endpoint: resolve the path's server before doing any work. A
    // disabled server is treated as absent — not served, same as the
    // aggregated endpoint skipping it (and same as `/a2a/:agent`).
    if let Some(server) = scope {
        let known = snapshot
            .mcp_servers
            .get_by_name(server)
            .is_some_and(|entry| entry.value.enabled);
        if !known {
            return (
                StatusCode::NOT_FOUND,
                format!("unknown MCP server: {server}"),
            )
                .into_response();
        }
    }

    // Buffer the body so the JSON-RPC method can be inspected, then rebuilt for
    // the gateway. The global body-limit layer has already capped the size.
    let (parts, body) = request.into_parts();
    let bytes = match to_bytes(
        body,
        crate::error::body_read_cap(state.request_body_limit_bytes),
    )
    .await
    {
        Ok(bytes) => bytes,
        // A cap hit is a 413 in the standard envelope — consistent with
        // what the Content-Length middleware already answers on this
        // route; anything else reading the body is a client fault.
        Err(err) if crate::error::is_length_limit_error(&err) => {
            return crate::error::ProxyError::RequestTooLarge {
                limit_bytes: state.request_body_limit_bytes,
            }
            .into_response();
        }
        Err(_) => return (StatusCode::BAD_REQUEST, "invalid request body").into_response(),
    };

    let peek = serde_json::from_slice::<JsonRpcPeek>(&bytes).ok();
    let is_tool_call = peek.as_ref().and_then(|p| p.method.as_deref()) == Some("tools/call");
    // Resolve the called (server, tool) up front, owned, so it survives the
    // body being consumed when the request is rebuilt. Aggregated: split the
    // namespaced name. Scoped: the server comes from the path and the name is
    // the upstream's original one — stripped through the same primitive the
    // gateway parses with (`strip_server_prefix`), so quota and usage
    // attribute the same tool for both spellings and can never drift from
    // what actually dispatches.
    let (mcp_server, mcp_tool) = if is_tool_call {
        let name = peek
            .as_ref()
            .and_then(|p| p.params.as_ref())
            .and_then(|p| p.name.as_deref())
            .unwrap_or_default();
        match scope {
            Some(server) => {
                let bare = aisix_mcp::strip_server_prefix(server, name).unwrap_or(name);
                (server.to_string(), bare.to_string())
            }
            None => name
                .split_once(aisix_mcp::TOOL_NAMESPACE_SEPARATOR)
                .map(|(server, tool)| (server.to_string(), tool.to_string()))
                .unwrap_or_default(),
        }
    } else {
        (String::new(), String::new())
    };

    // Reuse the LLM path's rate-limit + budget gate on the unit of work, plus
    // the key's own limit for the MCP server this tool belongs to
    // (AISIX-Cloud#1079). The reservation is held for the duration of the call
    // and dropped after (no tokens to commit — a tool call carries no token
    // cost), which releases the concurrency slot. On 429 / budget-exceeded this
    // returns before any upstream is contacted — and the rejected call is still
    // recorded.
    let _reservation = if is_tool_call {
        match crate::quota::enforce_mcp(state, &auth, &mcp_server).await {
            Ok(reservation) => Some(reservation),
            Err(err) => {
                let response = err.into_response();
                emit_tool_call_usage(
                    state,
                    &auth,
                    request_id,
                    &mcp_server,
                    &mcp_tool,
                    response.status().as_u16(),
                    Duration::ZERO,
                    false,
                    Vec::new(),
                );
                return response;
            }
        }
    } else {
        None
    };

    // Resolve the guardrail chain once and run BOTH directions through the SAME
    // chain as LLM traffic: the tool arguments (input) before the call, and the
    // tool result (output) after. MCP has no model, so an empty `model_id`
    // matches env / api-key / team-scoped guardrails, never a Model-scoped one.
    // An empty chain short-circuits, keeping the no-guardrail path cheap (and
    // skipping the response buffering the output check needs).
    let rpc_id = peek.as_ref().and_then(|p| p.id.clone());
    let guardrail_chain = is_tool_call
        .then(|| {
            let ctx = aisix_guardrails::RequestContext {
                model_id: "",
                api_key_id: &auth.entry.id,
                team_id: auth.key().team_id.as_deref(),
            };
            state.guardrail_index.resolve(&ctx)
        })
        .filter(|chain| !chain.is_empty());

    // Input guardrails: scan the tool arguments.
    let mut monitor_hits: Vec<aisix_core::GuardrailMonitorHit> = Vec::new();
    if let Some(chain) = &guardrail_chain {
        let args_text = peek
            .as_ref()
            .and_then(|p| p.params.as_ref())
            .and_then(|p| p.arguments.as_ref())
            .map(|args| args.to_string())
            .unwrap_or_default();
        let chat =
            aisix_gateway::ChatFormat::new("", vec![aisix_gateway::ChatMessage::user(args_text)]);
        let (verdict, hits) = aisix_guardrails::Guardrail::check_input_observed(chain, &chat).await;
        monitor_hits.extend(hits);
        if let aisix_guardrails::GuardrailVerdict::Block {
            reason,
            guardrail_name,
        } = verdict
        {
            tracing::warn!(
                guardrail_hook = "input",
                tool = %mcp_tool,
                reason = %reason,
                "guardrail blocked MCP tool call"
            );
            emit_tool_call_usage(
                state,
                &auth,
                request_id,
                &mcp_server,
                &mcp_tool,
                StatusCode::OK.as_u16(),
                Duration::ZERO,
                true,
                monitor_hits,
            );
            return jsonrpc_guardrail_block(rpc_id, "tool call", guardrail_name.as_deref());
        }
    }

    // Scope the gateway to the tools this caller's key permits — resolved
    // from the key together with the environment/team MCP access policies —
    // so MCP tool access is governed by the same key object as LLM access.
    let acl = aisix_mcp::ToolAcl::resolve(&snapshot, auth.key());
    let gateway = match scope {
        // Same snapshot as the resolution above, so the entry is still there.
        Some(server) => match aisix_mcp::McpGateway::from_snapshot_scoped(&snapshot, server) {
            Some(gateway) => gateway,
            None => {
                return (
                    StatusCode::NOT_FOUND,
                    format!("unknown MCP server: {server}"),
                )
                    .into_response()
            }
        },
        None => aisix_mcp::McpGateway::from_snapshot(&snapshot),
    }
    .with_tool_acl(acl);
    let service = aisix_mcp::streamable_http_service(gateway);
    let request = Request::from_parts(parts, Body::from(bytes));
    // `StreamableHttpService` is a tower service that dispatches on method and
    // never fails (`Error = Infallible`); map its boxed body back to axum's.
    let started = Instant::now();
    let response = match service.oneshot(request).await {
        Ok(response) => response.map(Body::new),
        Err(infallible) => match infallible {},
    };
    let latency = started.elapsed();

    // Output guardrails: scan the tool result before returning it. The response
    // body is only buffered when a guardrail chain is attached.
    let response = if let Some(chain) = &guardrail_chain {
        let (resp_parts, resp_body) = response.into_parts();
        let resp_bytes = match to_bytes(
            resp_body,
            crate::error::body_read_cap(state.request_body_limit_bytes),
        )
        .await
        {
            Ok(bytes) => bytes,
            Err(_) => {
                return (StatusCode::BAD_GATEWAY, "invalid upstream response").into_response()
            }
        };
        if let Some(guardrail_name) =
            output_guardrail_block(chain, &resp_bytes, &mcp_tool, &mut monitor_hits).await
        {
            emit_tool_call_usage(
                state,
                &auth,
                request_id,
                &mcp_server,
                &mcp_tool,
                StatusCode::OK.as_u16(),
                latency,
                true,
                monitor_hits,
            );
            return jsonrpc_guardrail_block(rpc_id, "tool result", guardrail_name.as_deref());
        }
        Response::from_parts(resp_parts, Body::from(resp_bytes))
    } else {
        response
    };

    if is_tool_call {
        emit_tool_call_usage(
            state,
            &auth,
            request_id,
            &mcp_server,
            &mcp_tool,
            response.status().as_u16(),
            latency,
            false,
            monitor_hits,
        );
    }
    response
}

/// Run the output guardrail chain over an MCP tool result. Returns `Some(_)` to
/// block — the inner value is the firing guardrail's name, or `None` for a
/// fail-closed block on a body that cannot be parsed — and `None` to allow. The
/// tool result's text is fed to `check_output` as assistant text, the same hook
/// the LLM response path uses; a protocol-level error envelope (no `result`) has
/// nothing to scan and is allowed.
async fn output_guardrail_block(
    chain: &aisix_guardrails::GuardrailChain,
    response_bytes: &[u8],
    tool: &str,
    monitor_hits: &mut Vec<aisix_core::GuardrailMonitorHit>,
) -> Option<Option<String>> {
    // Fail closed on an unparseable body. The `/mcp` gateway is configured
    // `json_response = true`, so a `tools/call` returns a single
    // `application/json` object; a body that does not parse (e.g. if that ever
    // regressed to SSE framing) must not slip an unscanned tool result past the
    // guardrail — block rather than allow.
    let value: serde_json::Value = match serde_json::from_slice(response_bytes) {
        Ok(value) => value,
        Err(_) => return Some(None),
    };
    // A protocol-level error envelope (no `result`) has no tool output to scan.
    let result = value.get("result")?;
    // Scan the client-visible tool text — the `text`-type content blocks the
    // result carries — not the serialized JSON envelope. This keeps MCP output
    // and LLM output on the same representation: a keyword guardrail sees the
    // decoded prose, so envelope field names (`content`, `type`, `text`) can't
    // trip a false positive, and escaped characters can't hide blocked content.
    // Fall back to the whole serialized result for non-standard shapes so
    // nothing escapes inspection.
    let result_text = result
        .get("content")
        .and_then(|c| c.as_array())
        .map(|blocks| {
            blocks
                .iter()
                .filter(|b| b.get("type").and_then(|t| t.as_str()) == Some("text"))
                .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join("\n")
        })
        .filter(|text| !text.is_empty())
        .unwrap_or_else(|| result.to_string());
    let resp = aisix_gateway::ChatResponse {
        id: String::new(),
        model: String::new(),
        message: aisix_gateway::ChatMessage::assistant(result_text),
        finish_reason: aisix_gateway::FinishReason::Stop,
        usage: aisix_gateway::UsageStats::new(0, 0),
    };
    let (verdict, hits) = aisix_guardrails::Guardrail::check_output_observed(chain, &resp).await;
    monitor_hits.extend(hits);
    match verdict {
        aisix_guardrails::GuardrailVerdict::Block {
            reason,
            guardrail_name,
        } => {
            tracing::warn!(
                guardrail_hook = "output",
                tool = %tool,
                reason = %reason,
                "guardrail blocked MCP tool result"
            );
            Some(guardrail_name)
        }
        _ => None,
    }
}

/// Emit a usage event for a single MCP tool call into the same sink as LLM
/// usage. MCP calls carry no token cost yet, so token/cost fields stay zero;
/// the event records who called which tool, the outcome, and the latency.
#[allow(clippy::too_many_arguments)]
fn emit_tool_call_usage(
    state: &ProxyState,
    auth: &AuthenticatedKey,
    request_id: &str,
    mcp_server: &str,
    mcp_tool: &str,
    status_code: u16,
    latency: Duration,
    guardrail_blocked: bool,
    guardrail_monitor_hits: Vec<aisix_core::GuardrailMonitorHit>,
) {
    let event = UsageEvent {
        request_id: request_id.to_string(),
        occurred_at: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        api_key_id: auth.entry.id.clone(),
        status_code,
        // Single-attempt endpoint: the attempt spans the whole request, so
        // the upstream figure and what the caller waited for coincide.
        upstream_latency_ms: latency.as_millis().min(u32::MAX as u128) as u32,
        downstream_latency_ms: latency.as_millis().min(u32::MAX as u128) as u32,
        inbound_protocol: "mcp".to_string(),
        mcp_server_name: mcp_server.to_string(),
        mcp_tool_name: mcp_tool.to_string(),
        guardrail_blocked,
        guardrail_monitor_hits,
        ..Default::default()
    };
    state.usage_sink.try_emit("mcp", event.clone());
    // #698: fan the event out to the per-env OTLP/SLS/Datadog exporters like
    // every other emitter — pre-fix MCP usage reached only the CP sink, so
    // exporters never saw /mcp traffic. No content capture (tool args/results
    // are a separate surface from prompt/response).
    let snap = state.snapshot.load();
    let exporters = snap.observability_exporters.entries();
    state
        .otlp_fan_out
        .fan_out(&event, None, exporters.iter().map(|e| &e.value));
}

/// Build the MCP-native response for a guardrail block: a JSON-RPC error
/// echoing the request id, served as HTTP 200 with a JSON body (the MCP
/// Streamable HTTP shape). Both the input and output hooks funnel through here;
/// `side` (`"tool call"` for input arguments, `"tool result"` for output)
/// selects the caller-visible wording. Unlike the LLM path's 422, an MCP client
/// expects a JSON-RPC envelope, so the block surfaces as an error it can handle.
fn jsonrpc_guardrail_block(
    id: Option<serde_json::Value>,
    side: &str,
    guardrail_name: Option<&str>,
) -> Response {
    let message = crate::error::guardrail_block_message(side, guardrail_name);
    let body = serde_json::json!({
        "jsonrpc": "2.0",
        "id": id.unwrap_or(serde_json::Value::Null),
        "error": { "code": -32600, "message": message }
    });
    (
        StatusCode::OK,
        [(axum::http::header::CONTENT_TYPE, "application/json")],
        body.to_string(),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build_router;
    use aisix_core::{AisixSnapshot, ApiKey, ProxyConfig, ResourceEntry, SnapshotHandle};
    use axum::body::Body;
    use axum::http::{Request as HttpRequest, StatusCode};
    use std::sync::Arc;

    fn cfg() -> ProxyConfig {
        ProxyConfig {
            addr: "127.0.0.1:0".into(),
            request_body_limit_bytes: 1_048_576,
            real_ip: Default::default(),
            tls: None,
        }
    }

    const TOKEN: &str = "sk-mcp-endpoint-test";

    /// A snapshot carrying one valid API key (and no MCP servers — the MCP
    /// `initialize` handshake is answered by the gateway itself, no upstream
    /// needed).
    fn snapshot_with_key() -> AisixSnapshot {
        let key_hash = ApiKey::hash_bearer(TOKEN);
        let apikey: ApiKey = serde_json::from_value(serde_json::json!({
            "key_hash": key_hash,
            "allowed_models": ["*"],
        }))
        .expect("valid apikey");
        let snapshot = AisixSnapshot::new();
        snapshot
            .apikeys
            .insert(ResourceEntry::new("ak-1", apikey, 1));
        snapshot
    }

    fn router_with(snapshot: AisixSnapshot) -> axum::Router {
        let handle = SnapshotHandle::new(snapshot);
        let hub = Arc::new(aisix_gateway::Hub::new());
        build_router(ProxyState::new(handle, hub, &cfg()).without_cache())
    }

    /// A minimal MCP `initialize` request body + the headers the Streamable
    /// HTTP transport requires (Accept must list both content types).
    fn initialize_request(auth: Option<&str>) -> HttpRequest<Body> {
        let body = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": { "name": "endpoint-test", "version": "0.1" }
            }
        });
        // A non-loopback Host on purpose: proves the gateway accepts the
        // deployment's real DNS name (rmcp's default Host allowlist is disabled
        // for this key-authenticated endpoint).
        let mut builder = HttpRequest::post("/mcp")
            .header("host", "mcp.aisix.example.com")
            .header("content-type", "application/json")
            .header("accept", "application/json, text/event-stream");
        if let Some(token) = auth {
            builder = builder.header("authorization", format!("Bearer {token}"));
        }
        builder.body(Body::from(body.to_string())).unwrap()
    }

    /// A snapshot whose key carries an inline `rate_limit` of `rpm` requests
    /// per minute and may call every tool.
    fn snapshot_with_rate_limited_key(rpm: u32) -> AisixSnapshot {
        let key_hash = ApiKey::hash_bearer(TOKEN);
        let apikey: ApiKey = serde_json::from_value(serde_json::json!({
            "key_hash": key_hash,
            "allowed_models": ["*"],
            "allowed_tools": ["*"],
            "rate_limit": { "rpm": rpm },
        }))
        .expect("valid apikey");
        let snapshot = AisixSnapshot::new();
        snapshot
            .apikeys
            .insert(ResourceEntry::new("ak-1", apikey, 1));
        snapshot
    }

    /// A JSON-RPC request to `/mcp` for `method`, authenticated with `TOKEN`.
    fn mcp_request(method: &str, params: serde_json::Value) -> HttpRequest<Body> {
        mcp_request_with_id(serde_json::json!(1), method, params)
    }

    /// As [`mcp_request`], but with an explicit JSON-RPC `id` so a test can
    /// assert the response echoes the request's id rather than a constant.
    fn mcp_request_with_id(
        id: serde_json::Value,
        method: &str,
        params: serde_json::Value,
    ) -> HttpRequest<Body> {
        let body = serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params
        });
        HttpRequest::post("/mcp")
            .header("host", "mcp.aisix.example.com")
            .header("content-type", "application/json")
            .header("accept", "application/json, text/event-stream")
            .header("authorization", format!("Bearer {TOKEN}"))
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    fn tools_call_request() -> HttpRequest<Body> {
        mcp_request(
            "tools/call",
            serde_json::json!({ "name": "ghost__tool", "arguments": {} }),
        )
    }

    /// A `tools/call` for `<server>__tool`, authenticated with `token`.
    fn tools_call_on(token: &str, server: &str) -> HttpRequest<Body> {
        let body = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": { "name": format!("{server}__tool"), "arguments": {} }
        });
        HttpRequest::post("/mcp")
            .header("host", "mcp.aisix.example.com")
            .header("content-type", "application/json")
            .header("accept", "application/json, text/event-stream")
            .header("authorization", format!("Bearer {token}"))
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    /// A snapshot carrying one key per `(id, token, mcp_rate_limits)` triple.
    /// No key-level `rate_limit`, so any 429 can only come from the
    /// per-MCP-server layer.
    fn snapshot_with_mcp_server_limits(keys: &[(&str, &str, serde_json::Value)]) -> AisixSnapshot {
        let snapshot = AisixSnapshot::new();
        for (id, token, limits) in keys {
            let apikey: ApiKey = serde_json::from_value(serde_json::json!({
                "key_hash": ApiKey::hash_bearer(token),
                "allowed_models": ["*"],
                "allowed_tools": ["*"],
                "mcp_rate_limits": limits,
            }))
            .expect("valid apikey");
            snapshot.apikeys.insert(ResourceEntry::new(*id, apikey, 1));
        }
        snapshot
    }

    #[tokio::test]
    async fn mcp_server_rate_limit_counts_per_server() {
        // The key may call `alpha` once a minute; `beta` is uncapped.
        let router = router_with(snapshot_with_mcp_server_limits(&[(
            "ak-1",
            TOKEN,
            serde_json::json!({ "alpha": { "rpm": 1 } }),
        )]));

        let first = router
            .clone()
            .oneshot(tools_call_on(TOKEN, "alpha"))
            .await
            .expect("router responds");
        assert_ne!(
            first.status(),
            StatusCode::TOO_MANY_REQUESTS,
            "first alpha tool call should pass the gate"
        );

        let second = router
            .clone()
            .oneshot(tools_call_on(TOKEN, "alpha"))
            .await
            .expect("router responds");
        assert_eq!(
            second.status(),
            StatusCode::TOO_MANY_REQUESTS,
            "second alpha tool call in the window should be rate-limited"
        );

        // A server the key sets no limit for keeps its own (unlimited)
        // counter — alpha's burst must not spend beta's budget.
        for _ in 0..3 {
            let other = router
                .clone()
                .oneshot(tools_call_on(TOKEN, "beta"))
                .await
                .expect("router responds");
            assert_ne!(
                other.status(),
                StatusCode::TOO_MANY_REQUESTS,
                "an unlimited server must not be limited by another server's burst"
            );
        }
    }

    #[tokio::test]
    async fn mcp_server_rate_limit_counts_per_key() {
        // Two keys, each capped at one `alpha` call per minute. Exhausting
        // one must leave the other's quota intact.
        const OTHER_TOKEN: &str = "sk-mcp-endpoint-test-2";
        let limits = serde_json::json!({ "alpha": { "rpm": 1 } });
        let router = router_with(snapshot_with_mcp_server_limits(&[
            ("ak-1", TOKEN, limits.clone()),
            ("ak-2", OTHER_TOKEN, limits),
        ]));

        for _ in 0..2 {
            router
                .clone()
                .oneshot(tools_call_on(TOKEN, "alpha"))
                .await
                .expect("router responds");
        }
        let exhausted = router
            .clone()
            .oneshot(tools_call_on(TOKEN, "alpha"))
            .await
            .expect("router responds");
        assert_eq!(
            exhausted.status(),
            StatusCode::TOO_MANY_REQUESTS,
            "the first key should be at its alpha limit"
        );

        let other_key = router
            .oneshot(tools_call_on(OTHER_TOKEN, "alpha"))
            .await
            .expect("router responds");
        assert_ne!(
            other_key.status(),
            StatusCode::TOO_MANY_REQUESTS,
            "a second key must not be limited by the first key's burst"
        );
    }

    #[tokio::test]
    async fn rate_limit_applies_to_tool_calls_but_not_handshake() {
        // rpm=1: the key may make one tools/call per minute.
        let router = router_with(snapshot_with_rate_limited_key(1));

        // First tool call passes the rate gate (status is whatever the gateway
        // returns — there are no upstreams — but NOT 429).
        let first = router
            .clone()
            .oneshot(tools_call_request())
            .await
            .expect("router responds");
        assert_ne!(
            first.status(),
            StatusCode::TOO_MANY_REQUESTS,
            "first tool call should pass the rate gate"
        );

        // Second tool call within the same minute is rate-limited.
        let second = router
            .clone()
            .oneshot(tools_call_request())
            .await
            .expect("router responds");
        assert_eq!(
            second.status(),
            StatusCode::TOO_MANY_REQUESTS,
            "second tool call should be rate-limited"
        );

        // Neither handshake nor discovery is rate-limited, even with the key at
        // its tool-call limit — a client can always connect and enumerate.
        let handshake = router
            .clone()
            .oneshot(initialize_request(Some(TOKEN)))
            .await
            .expect("router responds");
        assert_ne!(
            handshake.status(),
            StatusCode::TOO_MANY_REQUESTS,
            "initialize must not be rate-limited"
        );

        let listed = router
            .oneshot(mcp_request("tools/list", serde_json::json!({})))
            .await
            .expect("router responds");
        assert_ne!(
            listed.status(),
            StatusCode::TOO_MANY_REQUESTS,
            "tools/list must not be rate-limited"
        );
    }

    /// Read a JSON-RPC response body (the endpoint is configured for JSON
    /// responses, not SSE).
    async fn body_json(resp: axum::response::Response) -> serde_json::Value {
        let bytes = to_bytes(resp.into_body(), usize::MAX).await.expect("body");
        serde_json::from_slice(&bytes).expect("JSON-RPC body")
    }

    #[tokio::test]
    async fn mcp_access_deny_mode_rejects_tool_calls_at_the_acl() {
        // The key's legacy allowlist grants everything, but its mcp_access
        // block says deny — the policy layer must win. The rejection is the
        // ACL's neutral "not available" (reached before upstream routing),
        // not the router's "unknown MCP server", which proves the endpoint
        // resolves the ACL from the key + policies rather than allowed_tools
        // alone.
        let key_hash = ApiKey::hash_bearer(TOKEN);
        let apikey: ApiKey = serde_json::from_value(serde_json::json!({
            "key_hash": key_hash,
            "allowed_models": ["*"],
            "allowed_tools": ["*"],
            "mcp_access": { "mode": "deny" },
        }))
        .expect("valid apikey");
        let snapshot = AisixSnapshot::new();
        snapshot
            .apikeys
            .insert(ResourceEntry::new("ak-1", apikey, 1));

        let router = router_with(snapshot);
        let resp = router
            .oneshot(tools_call_request())
            .await
            .expect("router responds");
        let body = body_json(resp).await;
        let message = body["error"]["message"].as_str().unwrap_or_default();
        assert!(
            message.contains("not available"),
            "deny-mode key must be rejected by the ACL, got: {body}"
        );
    }

    #[tokio::test]
    async fn env_policy_deny_overlays_legacy_key_at_the_endpoint() {
        // A legacy key (no mcp_access block) with a wildcard allowlist, plus
        // an env policy that denies exactly one tool: the denied tool is
        // rejected by the ACL while any other name still reaches routing
        // (and fails as "unknown MCP server" — no upstreams are registered).
        let key_hash = ApiKey::hash_bearer(TOKEN);
        let apikey: ApiKey = serde_json::from_value(serde_json::json!({
            "key_hash": key_hash,
            "allowed_models": ["*"],
            "allowed_tools": ["*"],
        }))
        .expect("valid apikey");
        let policy: aisix_core::models::McpPolicy = serde_json::from_value(serde_json::json!({
            "scope": "env",
            "mode": "none",
            "deny": ["ghost__tool"],
        }))
        .expect("valid policy");
        let snapshot = AisixSnapshot::new();
        snapshot
            .apikeys
            .insert(ResourceEntry::new("ak-1", apikey, 1));
        snapshot
            .mcp_policies
            .insert(ResourceEntry::new("p-env", policy, 1));

        let router = router_with(snapshot);

        let denied = router
            .clone()
            .oneshot(tools_call_request()) // calls ghost__tool
            .await
            .expect("router responds");
        let body = body_json(denied).await;
        let message = body["error"]["message"].as_str().unwrap_or_default();
        assert!(
            message.contains("not available"),
            "env-policy deny must subtract from a legacy key, got: {body}"
        );

        let other = router
            .oneshot(mcp_request(
                "tools/call",
                serde_json::json!({ "name": "other__tool", "arguments": {} }),
            ))
            .await
            .expect("router responds");
        let body = body_json(other).await;
        let message = body["error"]["message"].as_str().unwrap_or_default();
        assert!(
            message.contains("unknown MCP server"),
            "a non-denied tool must still pass the ACL for a wildcard legacy key, got: {body}"
        );
    }

    #[tokio::test]
    async fn rejects_request_without_api_key() {
        let router = router_with(snapshot_with_key());
        let resp = router
            .oneshot(initialize_request(None))
            .await
            .expect("router responds");
        assert_eq!(
            resp.status(),
            StatusCode::UNAUTHORIZED,
            "missing API key must be rejected at the /mcp edge"
        );
    }

    #[tokio::test]
    async fn rejects_request_with_invalid_api_key() {
        let router = router_with(snapshot_with_key());
        let resp = router
            .oneshot(initialize_request(Some("sk-wrong")))
            .await
            .expect("router responds");
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn auth_gates_non_post_methods() {
        // The route is `any(...)`, so every method must be auth-gated — a GET
        // with no key must 401 (not fall through to rmcp's 405).
        let router = router_with(snapshot_with_key());
        let req = HttpRequest::get("/mcp")
            .header("host", "mcp.aisix.example.com")
            .body(Body::empty())
            .unwrap();
        let resp = router.oneshot(req).await.expect("router responds");
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn trailing_slash_route_is_auth_gated() {
        let router = router_with(snapshot_with_key());
        let req = HttpRequest::post("/mcp/")
            .header("host", "mcp.aisix.example.com")
            .header("content-type", "application/json")
            .header("accept", "application/json, text/event-stream")
            .body(Body::from("{}"))
            .unwrap();
        let resp = router.oneshot(req).await.expect("router responds");
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn oversized_unauthenticated_body_is_limited_before_handler() {
        // A declared Content-Length over the cap is rejected (413) by the
        // body-limit layer, which wraps the route — before auth or the handler,
        // so an oversized unauthenticated body can't pin resources.
        let router = router_with(snapshot_with_key());
        let big = "a".repeat(1_048_577); // cfg() cap is 1 MiB
        let req = HttpRequest::post("/mcp")
            .header("host", "mcp.aisix.example.com")
            .header("content-type", "application/json")
            .header("content-length", big.len().to_string())
            .body(Body::from(big))
            .unwrap();
        let resp = router.oneshot(req).await.expect("router responds");
        let status = resp.status();
        assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE, "got {status}");
    }

    #[tokio::test]
    async fn chunked_oversized_body_returns_enveloped_413() {
        // No Content-Length: the middleware can't pre-check, so the
        // handler's own capped read fires. The length-limit error must
        // surface as the enveloped 413 — matching what the middleware
        // answers on this route — not the bare-400 "invalid request
        // body" it used to fold into.
        let router = router_with(snapshot_with_key());
        let chunk = vec![b'a'; 200 * 1024];
        let stream =
            futures::stream::iter((0..10).map(move |_| Ok::<_, std::io::Error>(chunk.clone())));
        let req = HttpRequest::post("/mcp")
            .header("host", "mcp.aisix.example.com")
            .header("content-type", "application/json")
            .header("authorization", format!("Bearer {TOKEN}"))
            .body(Body::from_stream(stream))
            .unwrap();
        let resp = router.oneshot(req).await.expect("router responds");
        assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
        let body = axum::body::to_bytes(resp.into_body(), 64 * 1024)
            .await
            .expect("read body");
        let v: serde_json::Value =
            serde_json::from_slice(&body).expect("413 must carry the JSON envelope");
        assert_eq!(v["error"]["type"], "invalid_request_error");
    }

    #[tokio::test]
    async fn authenticated_request_reaches_the_mcp_gateway() {
        let router = router_with(snapshot_with_key());
        let resp = router
            .oneshot(initialize_request(Some(TOKEN)))
            .await
            .expect("router responds");
        // Auth passed and the request was served by the MCP gateway (not a 401).
        let status = resp.status();
        let body = axum::body::to_bytes(resp.into_body(), 64 * 1024)
            .await
            .expect("read body");
        let text = String::from_utf8_lossy(&body);
        assert_eq!(
            status,
            StatusCode::OK,
            "a valid key should reach the gateway and complete the MCP initialize handshake; body: {text}"
        );
        assert!(
            text.contains("serverInfo") || text.contains("protocolVersion"),
            "initialize result should carry the server info, got: {text}"
        );
    }

    #[tokio::test]
    async fn emits_usage_event_for_tool_call_only() {
        use aisix_obs::{UsageEvent, UsageSink};

        let (tx, mut rx) = tokio::sync::mpsc::channel::<UsageEvent>(8);
        let handle = SnapshotHandle::new(snapshot_with_key());
        let hub = Arc::new(aisix_gateway::Hub::new());
        let state = ProxyState::new(handle, hub, &cfg())
            .without_cache()
            .with_usage_sink(UsageSink::new(tx));
        let router = build_router(state);

        // A tools/call emits one usage event into the same sink as LLM calls,
        // carrying the MCP attribution (server + tool, parsed from the
        // namespaced name `ghost__tool`).
        let _ = router
            .clone()
            .oneshot(tools_call_request())
            .await
            .expect("router responds");
        let event = rx
            .try_recv()
            .expect("a usage event was emitted for the tool call");
        assert_eq!(event.inbound_protocol, "mcp");
        assert_eq!(event.mcp_server_name, "ghost");
        assert_eq!(event.mcp_tool_name, "tool");
        assert_eq!(event.api_key_id, "ak-1");
        assert_eq!(event.prompt_tokens, 0, "MCP calls carry no token cost");
        assert!(
            rx.try_recv().is_err(),
            "exactly one usage event per tool call"
        );

        // The handshake does NOT emit a usage event.
        let _ = router
            .oneshot(initialize_request(Some(TOKEN)))
            .await
            .expect("router responds");
        assert!(
            rx.try_recv().is_err(),
            "initialize must not emit a usage event"
        );
    }

    #[tokio::test]
    async fn rate_limited_tool_call_still_emits_usage_event() {
        use aisix_obs::{UsageEvent, UsageSink};

        // rpm=1: the second tool call is rate-limited (429) but still recorded —
        // the reject path emits before returning.
        let (tx, mut rx) = tokio::sync::mpsc::channel::<UsageEvent>(8);
        let handle = SnapshotHandle::new(snapshot_with_rate_limited_key(1));
        let hub = Arc::new(aisix_gateway::Hub::new());
        let state = ProxyState::new(handle, hub, &cfg())
            .without_cache()
            .with_usage_sink(UsageSink::new(tx));
        let router = build_router(state);

        let _ = router
            .clone()
            .oneshot(tools_call_request())
            .await
            .expect("router responds");
        let _ = rx.try_recv().expect("first (allowed) call emits");

        let second = router
            .oneshot(tools_call_request())
            .await
            .expect("router responds");
        assert_eq!(second.status(), StatusCode::TOO_MANY_REQUESTS);
        let event = rx
            .try_recv()
            .expect("the rate-limited call is still recorded");
        assert_eq!(event.status_code, 429);
        assert_eq!(event.inbound_protocol, "mcp");
        assert_eq!(event.mcp_server_name, "ghost");
        assert_eq!(event.mcp_tool_name, "tool");
    }

    /// Seed an env-scoped guardrail (from its JSON) by RCU-inserting it + an
    /// attachment into the live snapshot handle.
    fn seed_guardrail(handle: &SnapshotHandle<AisixSnapshot>, guardrail_json: &str) {
        use aisix_core::models::{Guardrail, GuardrailAttachment};
        let guardrail: Guardrail = serde_json::from_str(guardrail_json).unwrap();
        let attachment: GuardrailAttachment =
            serde_json::from_str(r#"{"guardrail_id":"g1","scope_type":"env","priority":50}"#)
                .unwrap();
        handle.rcu(|snap| {
            let new = snap.clone();
            new.guardrails
                .insert(ResourceEntry::new("g1", guardrail.clone(), 1));
            new.guardrail_attachments
                .insert(ResourceEntry::new("att-g1", attachment.clone(), 1));
            new
        });
    }

    const INPUT_GUARD: &str = r#"{"name":"mcp-input-guard","kind":"keyword","patterns":[{"kind":"literal","value":"forbidden-token"}]}"#;
    const OUTPUT_GUARD: &str = r#"{"name":"mcp-output-guard","kind":"keyword","hook_point":"output","patterns":[{"kind":"literal","value":"forbidden-token"}]}"#;

    fn tools_call_with_args(arguments: serde_json::Value) -> HttpRequest<Body> {
        mcp_request(
            "tools/call",
            serde_json::json!({ "name": "ghost__tool", "arguments": arguments }),
        )
    }

    #[tokio::test]
    async fn input_guardrail_blocks_tool_call_with_forbidden_args() {
        let handle = SnapshotHandle::new(snapshot_with_key());
        let hub = Arc::new(aisix_gateway::Hub::new());
        let state = ProxyState::new(handle.clone(), hub, &cfg()).without_cache();
        let router = build_router(state);
        seed_guardrail(&handle, INPUT_GUARD);

        // Arguments carrying the forbidden token are blocked by the same
        // guardrail chain LLM input uses — surfaced as an MCP-native JSON-RPC
        // error (HTTP 200) before the gateway/upstream is reached. A distinctive
        // request id (7) proves the handler echoes the caller's id (not a
        // constant) through the block envelope both hooks funnel through.
        let blocked = router
            .clone()
            .oneshot(mcp_request_with_id(
                serde_json::json!(7),
                "tools/call",
                serde_json::json!({ "name": "ghost__tool", "arguments": { "q": "forbidden-token" } }),
            ))
            .await
            .expect("router responds");
        assert_eq!(blocked.status(), StatusCode::OK);
        let body = axum::body::to_bytes(blocked.into_body(), 64 * 1024)
            .await
            .expect("read body");
        let envelope: serde_json::Value =
            serde_json::from_slice(&body).expect("a JSON-RPC envelope");
        assert_eq!(envelope["jsonrpc"], "2.0");
        assert_eq!(
            envelope["id"],
            serde_json::json!(7),
            "the block must echo the request id"
        );
        assert_eq!(envelope["error"]["code"], -32600);
        assert!(
            envelope.get("result").is_none(),
            "a guardrail block carries no result"
        );
        assert!(
            envelope["error"]["message"]
                .as_str()
                .unwrap_or_default()
                .contains("content policy"),
            "expected a content-policy block message, got: {envelope}"
        );

        // Clean arguments are not blocked by the guardrail (the gateway may
        // still reject for other reasons, but not with a content-policy error).
        let clean = router
            .oneshot(tools_call_with_args(serde_json::json!({ "q": "hello" })))
            .await
            .expect("router responds");
        let clean_body = axum::body::to_bytes(clean.into_body(), 64 * 1024)
            .await
            .expect("read body");
        assert!(
            !String::from_utf8_lossy(&clean_body).contains("content policy"),
            "clean arguments must not be guardrail-blocked"
        );
    }

    #[tokio::test]
    async fn output_guardrail_blocks_tool_result_with_forbidden_text() {
        use aisix_guardrails::{LiveGuardrailIndex, RequestContext};

        // Build the env-scoped output guardrail chain the handler would resolve.
        let handle = SnapshotHandle::new(snapshot_with_key());
        seed_guardrail(&handle, OUTPUT_GUARD);
        let index = LiveGuardrailIndex::new(handle, None);
        let chain = index.resolve(&RequestContext {
            model_id: "",
            api_key_id: "ak-1",
            team_id: None,
        });
        assert!(
            !chain.is_empty(),
            "output guardrail should resolve at env scope"
        );

        // A tool result whose content carries the forbidden token is blocked.
        let blocked = br#"{"jsonrpc":"2.0","id":1,"result":{"content":[{"type":"text","text":"forbidden-token here"}]}}"#;
        assert!(
            output_guardrail_block(&chain, blocked, "echo", &mut Vec::new())
                .await
                .is_some(),
            "a result containing the forbidden token must be blocked"
        );

        // A clean result passes.
        let clean =
            br#"{"jsonrpc":"2.0","id":1,"result":{"content":[{"type":"text","text":"all good"}]}}"#;
        assert!(
            output_guardrail_block(&chain, clean, "echo", &mut Vec::new())
                .await
                .is_none(),
            "a clean result must not be blocked"
        );

        // An error response (no `result`) has nothing to scan.
        let errored = br#"{"jsonrpc":"2.0","id":1,"error":{"code":-32600,"message":"x"}}"#;
        assert!(
            output_guardrail_block(&chain, errored, "echo", &mut Vec::new())
                .await
                .is_none(),
            "an error response has no tool result to scan"
        );

        // A body that is not JSON at all (e.g. SSE framing from a config
        // regression) fails closed — block, never pass an unscanned result.
        let sse_body = b"event: message\ndata: {\"jsonrpc\":\"2.0\",\"result\":{}}\n\n";
        assert!(
            output_guardrail_block(&chain, sse_body, "echo", &mut Vec::new())
                .await
                .is_some(),
            "an unparseable response body must fail closed (block)"
        );
    }

    #[tokio::test]
    async fn output_guardrail_scans_decoded_text_not_envelope() {
        use aisix_guardrails::{LiveGuardrailIndex, RequestContext};

        // A guardrail matching a JSON envelope field name ("content").
        const FIELD_NAME_GUARD: &str = r#"{"name":"field-name-guard","kind":"keyword","hook_point":"output","patterns":[{"kind":"literal","value":"content"}]}"#;
        let handle = SnapshotHandle::new(snapshot_with_key());
        seed_guardrail(&handle, FIELD_NAME_GUARD);
        let chain = LiveGuardrailIndex::new(handle, None).resolve(&RequestContext {
            model_id: "",
            api_key_id: "ak-1",
            team_id: None,
        });

        // The envelope literally contains "content"/"type"/"text", but we scan
        // the decoded tool text ("hello world"), so the field name must NOT fire.
        let clean = br#"{"jsonrpc":"2.0","id":1,"result":{"content":[{"type":"text","text":"hello world"}]}}"#;
        assert!(
            output_guardrail_block(&chain, clean, "echo", &mut Vec::new())
                .await
                .is_none(),
            "scanning the decoded text must ignore envelope field names"
        );

        // When the decoded text itself carries the pattern the guardrail fires —
        // proving it is active here, not simply absent.
        let hit = br#"{"jsonrpc":"2.0","id":1,"result":{"content":[{"type":"text","text":"this has content in it"}]}}"#;
        assert!(
            output_guardrail_block(&chain, hit, "echo", &mut Vec::new())
                .await
                .is_some(),
            "decoded text containing the pattern must still block"
        );
    }

    #[tokio::test]
    async fn output_block_envelope_echoes_id_and_shape() {
        // Both hooks funnel the block through `jsonrpc_guardrail_block`; assert
        // the wire envelope directly so a regression that nulls the id or shifts
        // the code/status/content-type is caught without an rmcp upstream.
        let resp = jsonrpc_guardrail_block(
            Some(serde_json::json!(42)),
            "tool result",
            Some("mcp-output-guard"),
        );
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers()
                .get(axum::http::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok()),
            Some("application/json")
        );
        let body = axum::body::to_bytes(resp.into_body(), 64 * 1024)
            .await
            .expect("read body");
        let v: serde_json::Value = serde_json::from_slice(&body).expect("a JSON-RPC envelope");
        assert_eq!(v["jsonrpc"], "2.0");
        assert_eq!(
            v["id"],
            serde_json::json!(42),
            "the original JSON-RPC id must be echoed, not nulled"
        );
        assert_eq!(v["error"]["code"], -32600);
        assert!(
            v.get("result").is_none(),
            "a block envelope carries no result"
        );
        assert!(
            v["error"]["message"]
                .as_str()
                .unwrap_or_default()
                .contains("tool result blocked by content policy"),
            "expected the output-side wording, got: {v}"
        );
    }

    /// #698: a tool-call usage event must reach the per-env observability
    /// exporters via the OTLP fan-out — pre-fix MCP usage was emitted only
    /// into the CP sink, so exporters never saw /mcp traffic. Uses the ghost
    /// server (no upstream needed): the gateway's error reply still records
    /// the call.
    #[tokio::test]
    async fn tool_call_usage_fans_out_to_exporters_issue_698() {
        use wiremock::matchers::method;
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let collector = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200))
            .mount(&collector)
            .await;

        let snapshot = snapshot_with_key();
        let exporter: aisix_core::ObservabilityExporter =
            serde_json::from_value(serde_json::json!({
                "name": "mcp-exp",
                "enabled": true,
                "kind": "otlp_http",
                "endpoint": format!("{}/v1/traces", collector.uri()),
                "headers": {}
            }))
            .expect("valid exporter");
        snapshot
            .observability_exporters
            .insert(ResourceEntry::new("exp-1", exporter, 1));

        let handle = SnapshotHandle::new(snapshot);
        let hub = Arc::new(aisix_gateway::Hub::new());
        let router = build_router(ProxyState::new(handle, hub, &cfg()).without_cache());

        let resp = router
            .oneshot(tools_call_request())
            .await
            .expect("router responds");
        assert_eq!(resp.status(), StatusCode::OK);

        // The fan-out POST runs in a detached task — poll for it.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3);
        loop {
            let received = collector.received_requests().await.unwrap_or_default();
            if !received.is_empty() {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "the tool-call usage event never reached the OTLP exporter"
            );
            tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        }
    }

    // ── /mcp/{server} — the scoped, single-server endpoint ──

    /// Seed an MCP server row. The URL is unroutable on purpose: these tests
    /// pin routing, resolution and attribution, not upstream success (that is
    /// covered by the aisix-mcp integration tests and the e2e suite).
    fn insert_mcp_server(snapshot: &AisixSnapshot, id: &str, name: &str, enabled: bool) {
        let server: aisix_core::McpServer = serde_json::from_value(serde_json::json!({
            "display_name": name,
            "url": "http://127.0.0.1:9/mcp",
            "enabled": enabled,
        }))
        .expect("valid mcp server");
        snapshot
            .mcp_servers
            .insert(ResourceEntry::new(id, server, 1));
    }

    /// A JSON-RPC request to `/mcp/{server}`, optionally authenticated.
    fn scoped_request(
        server: &str,
        auth: Option<&str>,
        method: &str,
        params: serde_json::Value,
    ) -> HttpRequest<Body> {
        let body = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params
        });
        let mut builder = HttpRequest::post(format!("/mcp/{server}"))
            .header("host", "mcp.aisix.example.com")
            .header("content-type", "application/json")
            .header("accept", "application/json, text/event-stream");
        if let Some(token) = auth {
            builder = builder.header("authorization", format!("Bearer {token}"));
        }
        builder.body(Body::from(body.to_string())).unwrap()
    }

    fn scoped_tools_call(server: &str, name: &str) -> HttpRequest<Body> {
        scoped_request(
            server,
            Some(TOKEN),
            "tools/call",
            serde_json::json!({ "name": name, "arguments": {} }),
        )
    }

    /// A snapshot with one enabled server `alpha` and a key that may call
    /// every tool.
    fn scoped_snapshot() -> AisixSnapshot {
        let apikey: ApiKey = serde_json::from_value(serde_json::json!({
            "key_hash": ApiKey::hash_bearer(TOKEN),
            "allowed_models": ["*"],
            "allowed_tools": ["*"],
        }))
        .expect("valid apikey");
        let snapshot = AisixSnapshot::new();
        snapshot
            .apikeys
            .insert(ResourceEntry::new("ak-1", apikey, 1));
        insert_mcp_server(&snapshot, "mcp-1", "alpha", true);
        snapshot
    }

    #[tokio::test]
    async fn scoped_endpoint_auth_precedes_server_resolution() {
        // No credentials → 401, even for a server that does not exist: an
        // unauthenticated caller must not learn which servers are registered.
        let router = router_with(snapshot_with_key());
        let response = router
            .oneshot(scoped_request(
                "ghost",
                None,
                "initialize",
                serde_json::json!({}),
            ))
            .await
            .expect("router responds");
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn scoped_endpoint_unknown_or_disabled_server_is_404() {
        // Unregistered server.
        let router = router_with(scoped_snapshot());
        let response = router
            .oneshot(scoped_tools_call("ghost", "tool"))
            .await
            .expect("router responds");
        assert_eq!(response.status(), StatusCode::NOT_FOUND);

        // Registered but disabled server: treated as absent, no fallback to
        // the aggregated surface.
        let snapshot = scoped_snapshot();
        insert_mcp_server(&snapshot, "mcp-2", "dark", false);
        let router = router_with(snapshot);
        let response = router
            .oneshot(scoped_tools_call("dark", "tool"))
            .await
            .expect("router responds");
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn scoped_tool_call_attributes_server_from_path() {
        use aisix_obs::{UsageEvent, UsageSink};

        let (tx, mut rx) = tokio::sync::mpsc::channel::<UsageEvent>(8);
        let handle = SnapshotHandle::new(scoped_snapshot());
        let hub = Arc::new(aisix_gateway::Hub::new());
        let state = ProxyState::new(handle, hub, &cfg())
            .without_cache()
            .with_usage_sink(UsageSink::new(tx));
        let router = build_router(state);

        // A bare (original) tool name: the server comes from the path, not
        // from a namespace prefix inside the name.
        let _ = router
            .clone()
            .oneshot(scoped_tools_call("alpha", "tool"))
            .await
            .expect("router responds");
        let event = rx.try_recv().expect("usage event for the bare-name call");
        assert_eq!(event.inbound_protocol, "mcp");
        assert_eq!(event.mcp_server_name, "alpha");
        assert_eq!(event.mcp_tool_name, "tool");

        // The namespaced spelling attributes identically — same tool, same
        // server — so quota and usage cannot be split by client spelling.
        let _ = router
            .oneshot(scoped_tools_call("alpha", "alpha__tool"))
            .await
            .expect("router responds");
        let event = rx.try_recv().expect("usage event for the prefixed call");
        assert_eq!(event.mcp_server_name, "alpha");
        assert_eq!(event.mcp_tool_name, "tool");
    }

    #[tokio::test]
    async fn scoped_per_server_rate_limit_keys_on_path_server() {
        // The key may call `alpha` once a minute. On the scoped endpoint the
        // limit must key on the path's server even for bare tool names.
        let snapshot = snapshot_with_mcp_server_limits(&[(
            "ak-1",
            TOKEN,
            serde_json::json!({ "alpha": { "rpm": 1 } }),
        )]);
        insert_mcp_server(&snapshot, "mcp-1", "alpha", true);
        let router = router_with(snapshot);

        let first = router
            .clone()
            .oneshot(scoped_tools_call("alpha", "tool"))
            .await
            .expect("router responds");
        assert_eq!(first.status(), StatusCode::OK);

        let second = router
            .clone()
            .oneshot(scoped_tools_call("alpha", "tool"))
            .await
            .expect("router responds");
        assert_eq!(second.status(), StatusCode::TOO_MANY_REQUESTS);

        // The namespaced spelling shares the SAME bucket — a client cannot
        // double its per-server allowance by switching spellings.
        let prefixed = router
            .oneshot(scoped_tools_call("alpha", "alpha__tool"))
            .await
            .expect("router responds");
        assert_eq!(prefixed.status(), StatusCode::TOO_MANY_REQUESTS);
    }

    #[tokio::test]
    async fn scoped_and_aggregated_endpoints_share_the_per_server_bucket() {
        // rpm=1 for `alpha`: one call through the aggregated endpoint must
        // exhaust the allowance for the scoped endpoint too — switching
        // endpoints cannot double the limit.
        let snapshot = snapshot_with_mcp_server_limits(&[(
            "ak-1",
            TOKEN,
            serde_json::json!({ "alpha": { "rpm": 1 } }),
        )]);
        insert_mcp_server(&snapshot, "mcp-1", "alpha", true);
        let router = router_with(snapshot);

        let aggregated = router
            .clone()
            .oneshot(tools_call_on(TOKEN, "alpha"))
            .await
            .expect("router responds");
        assert_eq!(aggregated.status(), StatusCode::OK);

        let scoped = router
            .oneshot(scoped_tools_call("alpha", "tool"))
            .await
            .expect("router responds");
        assert_eq!(scoped.status(), StatusCode::TOO_MANY_REQUESTS);
    }
}
