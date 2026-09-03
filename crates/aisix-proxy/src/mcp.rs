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
//! the caller's API key: per-tool access control (the layered MCP ACL),
//! rate-limit + budget (`quota::enforce_mcp`, which adds the key's per-MCP-server
//! limit to the shared layers), guardrails on both the tool arguments (input)
//! and the tool result (output), and a usage event into the shared sink.

use std::time::{Duration, Instant};

use aisix_obs::{AccessLog, UsageEvent};
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
pub async fn mcp_endpoint(State(state): State<ProxyState>, request: Request) -> Response {
    serve(state, request, None).await
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
    crate::reject::AisixPath(server): crate::reject::AisixPath<String>,
    State(state): State<ProxyState>,
    request: Request,
) -> Response {
    serve(state, request, Some(server)).await
}

/// The resolved caller of a `/mcp` request.
struct McpCaller {
    auth: AuthenticatedKey,
    /// The anonymous entry's server allowlist, which caps what this
    /// caller may see and call. `None` when the caller authenticated —
    /// an authenticated principal is bounded by its own grant alone.
    anonymous_servers: Option<Vec<String>>,
}

/// Authenticate the caller of a `/mcp` entry.
///
/// Order matters, and it is the whole security argument of the feature:
///
/// 1. A request that OFFERS a credential is authenticated, full stop. A
///    bad, expired, disabled or malformed one is rejected — it never
///    falls through to the anonymous principal, which would silently
///    swap the caller's identity for a different grant.
/// 2. Only a request offering nothing at all consults the anonymous
///    configuration for the addressed entry.
/// 3. Anything else takes the standard extractor path, whose 401 (and
///    `missing_credentials` metric) is what the endpoint answered
///    before this feature existed.
///
/// Step 3 is also what keeps the registered server set invisible: the
/// scoped handler resolves the path's server only after this returns, so
/// an unknown server and a known-but-not-anonymous one both answer 401
/// to an anonymous prober, never 404.
async fn resolve_caller(
    state: &ProxyState,
    parts: &mut axum::http::request::Parts,
    scope: Option<&str>,
) -> Result<McpCaller, crate::error::ProxyError> {
    if !crate::auth::credential_offered(parts) {
        let snapshot = state.snapshot.load();
        if let Some(anon) = crate::mcp_auth::anonymous_entry(state, &snapshot, parts, scope) {
            // Publish the principal the way the extractor does, so the
            // extractors running after this one — `ClientContext` and
            // the `${request.api_key.*}` header templates — see the
            // caller without re-authenticating.
            parts.extensions.insert(anon.auth.entry.clone());
            return Ok(McpCaller {
                auth: anon.auth,
                anonymous_servers: Some(anon.servers),
            });
        }
    }
    let auth =
        <AuthenticatedKey as axum::extract::FromRequestParts<ProxyState>>::from_request_parts(
            parts, state,
        )
        .await?;
    Ok(McpCaller {
        auth,
        anonymous_servers: None,
    })
}

async fn serve(state: ProxyState, request: Request, scope: Option<String>) -> Response {
    // Authentication runs here rather than as an extractor because the
    // decision depends on which entry was addressed: a no-credential
    // request may be served anonymously on an entry configured for it
    // (AISIX-Cloud#1313). A rejection short-circuits WITHOUT an access
    // log or request metric, exactly as the extractor's 401 did before
    // — an internet-facing DP would otherwise drown in scanner probes
    // (AISIX-Cloud#1081); `aisix_auth_decisions_total` records it.
    let (mut parts, body) = request.into_parts();
    let caller = match resolve_caller(&state, &mut parts, scope.as_deref()).await {
        Ok(caller) => caller,
        Err(err) => return err.into_response(),
    };
    let McpCaller {
        auth,
        anonymous_servers,
    } = caller;
    let request = Request::from_parts(parts, body);
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
    // The request's trace bundle (AISIX-Cloud#1279), minted beside the id.
    let trace = request
        .extensions()
        .get::<std::sync::Arc<aisix_obs::RequestTraceBundle>>()
        .cloned();
    let api_key_id = auth.entry.id.clone();
    let method = request.method().clone();
    // `dispatch` takes the key by value; the terminal emit below still needs
    // the caller's team / user labels (the handle is an `Arc` clone).
    let caller_auth = auth.clone();

    let response = dispatch(
        auth,
        anonymous_servers.as_deref(),
        scope.as_deref(),
        &state,
        request,
        &request_id,
        trace.as_ref(),
    )
    .await;

    let elapsed = started.elapsed();
    let status = response.status().as_u16();
    // Bounded route template, mirroring `/a2a` (the per-request server is
    // on the usage event, not the access log).
    let endpoint = if scope.is_some() {
        "/mcp/{server}"
    } else {
        "/mcp"
    };
    AccessLog {
        method: method.as_str(),
        path: endpoint,
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
        provider_request_id: None,
        served_by_model: None,
        routing_attempt_count: None,
        routing_fallback_count: None,
    }
    .emit();
    crate::request_metrics::record(
        &state,
        endpoint,
        crate::request_metrics::Caller::new(&caller_auth),
        crate::request_metrics::Upstream {
            provider: "mcp",
            model: MCP_MODEL_LABEL,
            ..Default::default()
        },
        status,
        elapsed,
    );
    response
}

async fn dispatch(
    auth: AuthenticatedKey,
    anonymous_servers: Option<&[String]>,
    scope: Option<&str>,
    state: &ProxyState,
    request: Request,
    request_id: &str,
    trace: Option<&std::sync::Arc<aisix_obs::RequestTraceBundle>>,
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
    // `parts` is mutable so the input mask write-back below can refresh
    // Content-Length when it changes the body.
    let (mut parts, body) = request.into_parts();
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

    // Converge the accepted `MCP-Protocol-Version` set before any quota,
    // guardrail, or upstream work (AISIX-Cloud#1148). rmcp's own transport
    // check admits its whole hardcoded KNOWN_VERSIONS list — including
    // `2024-11-05`, whose HTTP+SSE transport this endpoint has never served —
    // and renders violations as a bare text/plain 400 outside the JSON-RPC
    // envelope. This gate rejects against the SAME list that `initialize`
    // negotiation and `server/discover` advertise, in the same envelope every
    // other synthesized `/mcp` error uses. An absent header passes: the
    // spec's backwards-compatibility rule (assume `2025-03-26`) applies
    // downstream.
    if let Some(response) = reject_unsupported_protocol_version(&parts.headers, peek.as_ref()) {
        return response;
    }

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
        match crate::quota::enforce_mcp(state, &snapshot, &auth, &mcp_server).await {
            Ok(reservation) => Some(reservation),
            Err(err) => {
                let response = err.into_response();
                emit_tool_call_usage(
                    state,
                    &snapshot,
                    &auth,
                    request_id,
                    &mcp_server,
                    &mcp_tool,
                    response.status().as_u16(),
                    Duration::ZERO,
                    false,
                    Vec::new(),
                    crate::redact::RedactionCounts::new(),
                    /* guardrail_chain */ None,
                    None,
                    trace,
                    /* dispatched */ false,
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
    // never matches a Model-scoped guardrail; the called server's id carries
    // the MCP-side dimension instead, so env / mcp-server / api-key / team
    // scopes all apply. An unregistered server name (a malformed namespaced
    // tool) leaves the id empty and simply matches no MCP-server scope.
    // An empty chain short-circuits, keeping the no-guardrail path cheap (and
    // skipping the response buffering the output check needs).
    let rpc_id = peek.as_ref().and_then(|p| p.id.clone());
    let guardrail_chain = is_tool_call
        .then(|| {
            let mcp_server_id = snapshot
                .mcp_servers
                .get_by_name(&mcp_server)
                .map(|entry| entry.id.clone())
                .unwrap_or_default();
            let ctx = aisix_guardrails::RequestContext {
                passthrough_route_id: "",
                model_id: "",
                mcp_server_id: &mcp_server_id,
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
        // Segment-moderating members (custom scripts, Bedrock ANONYMIZE,
        // Presidio, Lakera, Aliyun AI) are consulted through the segment
        // pass below instead — the same check/moderate split every LLM
        // family uses, so a member is never consulted (or billed) twice
        // per hook.
        let (verdict, hits) =
            aisix_guardrails::Guardrail::check_input_non_segment_observed(chain, &chat).await;
        monitor_hits.extend(hits);
        if let aisix_guardrails::GuardrailVerdict::Block {
            reason,
            guardrail_name,
            unavailable,
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
                &snapshot,
                &auth,
                request_id,
                &mcp_server,
                &mcp_tool,
                StatusCode::OK.as_u16(),
                Duration::ZERO,
                true,
                monitor_hits,
                crate::redact::RedactionCounts::new(),
                guardrail_chain.as_ref(),
                None,
                trace,
                /* dispatched */ false,
            );
            return jsonrpc_guardrail_block(
                rpc_id,
                "tool call",
                guardrail_name.as_deref(),
                unavailable.as_deref(),
            );
        }
    }

    // Content capture opt-in (AISIX-Cloud#1330): the largest cap any
    // full-content exporter requests, same gate as the LLM handlers.
    // Resolved before the write-backs so the response is buffered when
    // capture needs it even without a guardrail chain.
    let capture_cap = if is_tool_call {
        aisix_obs::content_capture_cap(
            snapshot
                .observability_exporters
                .entries()
                .iter()
                .map(|e| &e.value),
        )
    } else {
        None
    };

    // Input mask write-back (AISIX-Cloud#1330): rewrite the string leaves
    // under `params.arguments`, splicing on the raw bytes so every byte
    // outside a masked span reaches the gateway verbatim (`json_splice`).
    // Runs AFTER the block check — block rules judge the original text —
    // and only when the chain has a sync redactor (kind=pii mask rules).
    let mut redaction_counts = crate::redact::RedactionCounts::new();
    let bytes = match &guardrail_chain {
        Some(chain) if aisix_guardrails::Guardrail::redacts_input(chain) => {
            match rewrite_tool_arguments(chain, &bytes) {
                Ok(None) => bytes,
                Ok(Some((rewritten, counts))) => {
                    crate::redact::merge_counts(&mut redaction_counts, counts);
                    // The body length changed; the inner service must not
                    // trust a stale Content-Length.
                    parts.headers.insert(
                        axum::http::header::CONTENT_LENGTH,
                        axum::http::HeaderValue::from(rewritten.len()),
                    );
                    axum::body::Bytes::from(rewritten)
                }
                Err(err) => {
                    // Structurally impossible (the body already parsed as
                    // JSON for the peek) — fail closed rather than forward
                    // content the operator's mask policy should have hidden.
                    tracing::warn!(
                        tool = %mcp_tool,
                        error = %err,
                        "mcp input mask splice failed; blocking tool call",
                    );
                    emit_tool_call_usage(
                        state,
                        &snapshot,
                        &auth,
                        request_id,
                        &mcp_server,
                        &mcp_tool,
                        StatusCode::OK.as_u16(),
                        Duration::ZERO,
                        true,
                        monitor_hits,
                        crate::redact::RedactionCounts::new(),
                        guardrail_chain.as_ref(),
                        None,
                        trace,
                        /* dispatched */ false,
                    );
                    return jsonrpc_guardrail_block(
                        rpc_id,
                        "tool call",
                        None,
                        Some(crate::error::TAG_MASK_WRITEBACK_FAILED),
                    );
                }
            }
        }
        _ => bytes,
    };
    // Async segment-moderation pass over the same `params.arguments`
    // string leaves the sync write-back covers (#1363): every
    // segment-moderating member decides — and masks — through here. The
    // pass reuses the byte-splice walker, so the scan slots and the
    // write-back slots are the same set by construction (no fourth text
    // shape; the aisix#1027 scan/rewrite divergence is not widened).
    let bytes = match &guardrail_chain {
        Some(chain) if aisix_guardrails::Guardrail::moderates_segments(chain) => {
            match moderate_tool_arguments(chain, &bytes, &mut monitor_hits, &mut redaction_counts)
                .await
            {
                SegmentPassOutcome::Keep => bytes,
                SegmentPassOutcome::Rewritten(rewritten) => {
                    parts.headers.insert(
                        axum::http::header::CONTENT_LENGTH,
                        axum::http::HeaderValue::from(rewritten.len()),
                    );
                    axum::body::Bytes::from(rewritten)
                }
                SegmentPassOutcome::Block {
                    guardrail_name,
                    unavailable,
                } => {
                    emit_tool_call_usage(
                        state,
                        &snapshot,
                        &auth,
                        request_id,
                        &mcp_server,
                        &mcp_tool,
                        StatusCode::OK.as_u16(),
                        Duration::ZERO,
                        true,
                        monitor_hits,
                        redaction_counts,
                        guardrail_chain.as_ref(),
                        None,
                        trace,
                        /* dispatched */ false,
                    );
                    return jsonrpc_guardrail_block(
                        rpc_id,
                        "tool call",
                        guardrail_name.as_deref(),
                        unavailable.as_deref(),
                    );
                }
            }
        }
        _ => bytes,
    };
    // Post-mask by construction: cloned AFTER the write-backs above, so a
    // capturing exporter can never archive a value the mask removed
    // (same ordering rule as the LLM path — mask first, capture after).
    let captured_args = capture_cap.map(|_| bytes.clone());

    // Scope the gateway to the tools this caller's key permits — resolved
    // from the key together with the environment/team MCP access policies —
    // so MCP tool access is governed by the same key object as LLM access.
    let acl = {
        let resolved = aisix_mcp::ToolAcl::resolve(&snapshot, auth.key());
        // The anonymous allowlist is a CEILING on the bound principal,
        // not just the entry gate. Applied here — one layer on the ACL
        // both endpoints share — it constrains `tools/list` and
        // `tools/call` alike, so an anonymous caller cannot reach an
        // unlisted server by naming `<server>__<tool>` on the
        // aggregated endpoint while its scoped entry stays closed.
        match anonymous_servers {
            Some(servers) => resolved.narrowed_to_servers(servers),
            None => resolved,
        }
    };
    // The agent's own inbound headers, forwarded to every registered
    // server whose `forward_client_headers` admits them — an internal
    // server that authorizes on the end user's own credential rather than
    // on the gateway's. Every server forwards nothing by default.
    //
    // Read off `parts` rather than the request that is about to be rebuilt
    // below: only `content-length` was ever rewritten, and that header is
    // never forwarded.
    let client_headers = parts.headers.clone();
    let gateway = match scope {
        // Same snapshot as the resolution above, so the entry is still there.
        Some(server) => match aisix_mcp::McpGateway::from_snapshot_scoped_for_request(
            &snapshot,
            server,
            Some(&client_headers),
        ) {
            Some(gateway) => gateway,
            None => {
                return (
                    StatusCode::NOT_FOUND,
                    format!("unknown MCP server: {server}"),
                )
                    .into_response()
            }
        },
        None => aisix_mcp::McpGateway::from_snapshot_for_request(&snapshot, Some(&client_headers)),
    }
    .with_tool_acl(acl);
    // The deployment's body cap replaces rmcp's own 4 MiB default inside
    // the service; the proxy-level read above already enforced the same
    // limit, so the two layers can never disagree.
    let service = aisix_mcp::streamable_http_service(gateway, state.request_body_limit_bytes);
    let request = Request::from_parts(parts, Body::from(bytes));
    // `StreamableHttpService` is a tower service that dispatches on method and
    // never fails (`Error = Infallible`); map its boxed body back to axum's.
    let started = Instant::now();
    let response = match service.oneshot(request).await {
        Ok(response) => response.map(Body::new),
        Err(infallible) => match infallible {},
    };
    let latency = started.elapsed();

    // Output guardrails + mask write-back: scan the tool result before
    // returning it, rewriting masked spans in place. The response body is
    // buffered when a guardrail chain is attached OR a full-content
    // exporter wants the result captured.
    let mut captured_result: Option<axum::body::Bytes> = None;
    let response = if guardrail_chain.is_some() || capture_cap.is_some() {
        let (mut resp_parts, resp_body) = response.into_parts();
        let mut resp_bytes = match to_bytes(
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
        if let Some(chain) = &guardrail_chain {
            match apply_output_guardrails(chain, &resp_bytes, &mcp_tool, &mut monitor_hits).await {
                ToolResultOutcome::Block {
                    guardrail_name,
                    unavailable,
                } => {
                    emit_tool_call_usage(
                        state,
                        &snapshot,
                        &auth,
                        request_id,
                        &mcp_server,
                        &mcp_tool,
                        StatusCode::OK.as_u16(),
                        latency,
                        true,
                        monitor_hits,
                        redaction_counts,
                        guardrail_chain.as_ref(),
                        None,
                        trace,
                        /* dispatched */ true,
                    );
                    return jsonrpc_guardrail_block(
                        rpc_id,
                        "tool result",
                        guardrail_name.as_deref(),
                        unavailable.as_deref(),
                    );
                }
                ToolResultOutcome::Allow(Some((rewritten, counts))) => {
                    crate::redact::merge_counts(&mut redaction_counts, counts);
                    resp_parts.headers.insert(
                        axum::http::header::CONTENT_LENGTH,
                        axum::http::HeaderValue::from(rewritten.len()),
                    );
                    resp_bytes = axum::body::Bytes::from(rewritten);
                }
                ToolResultOutcome::Allow(None) => {}
            }
        }
        // Post-mask by construction — cloned after the write-back above.
        captured_result = capture_cap.map(|_| resp_bytes.clone());
        Response::from_parts(resp_parts, Body::from(resp_bytes))
    } else {
        response
    };

    if is_tool_call {
        let capture = capture_cap.map(|cap| {
            tool_call_capture(cap, captured_args.as_deref(), captured_result.as_deref())
        });
        emit_tool_call_usage(
            state,
            &snapshot,
            &auth,
            request_id,
            &mcp_server,
            &mcp_tool,
            response.status().as_u16(),
            latency,
            false,
            monitor_hits,
            redaction_counts,
            guardrail_chain.as_ref(),
            capture.as_ref(),
            trace,
            /* dispatched */ true,
        );
    }
    response
}

/// Splice-rewrite the string leaves under `params.arguments` of a
/// `tools/call` body through the chain's input redactor (AISIX-Cloud#1330).
/// `Ok(None)` = nothing masked, keep the original bytes.
fn rewrite_tool_arguments(
    chain: &aisix_guardrails::GuardrailChain,
    body: &[u8],
) -> Result<Option<(Vec<u8>, crate::redact::RedactionCounts)>, crate::json_splice::SpliceError> {
    let mut counts = crate::redact::RedactionCounts::new();
    let out = crate::json_splice::rewrite_string_values(body, tool_argument_path, |text| {
        aisix_guardrails::Guardrail::redact_input_text(chain, text).map(|r| {
            crate::redact::merge_counts(&mut counts, r.counts);
            r.text
        })
    })?;
    Ok(out.map(|bytes| (bytes, counts)))
}

/// Path predicate for the `params.arguments` string leaves — shared by
/// the sync write-back and both walks of the segment pass, so the scan
/// slots and the write-back slots can never drift.
fn tool_argument_path(path: &[crate::json_splice::PathSeg]) -> bool {
    path.first().is_some_and(|s| s.is_key("params"))
        && path.get(1).is_some_and(|s| s.is_key("arguments"))
}

/// Path predicate for the tool-result surfaces:
/// `result.content[i].{text,description,title}`,
/// `result.content[i].resource.text`, and every string leaf under
/// `result.structuredContent`. (`name`/`uri` stay untouched — they
/// address a resource.)
fn tool_result_path(path: &[crate::json_splice::PathSeg]) -> bool {
    if !path.first().is_some_and(|s| s.is_key("result")) {
        return false;
    }
    match path.get(1) {
        Some(seg) if seg.is_key("structuredContent") => true,
        Some(seg) if seg.is_key("content") => {
            matches!(path.get(2), Some(crate::json_splice::PathSeg::Index(_)))
                && ((path.len() == 4
                    && (path[3].is_key("text")
                        || path[3].is_key("description")
                        || path[3].is_key("title")))
                    || (path.len() == 5 && path[3].is_key("resource") && path[4].is_key("text")))
        }
        _ => false,
    }
}

/// Outcome of one async segment-moderation pass over a JSON-RPC body.
enum SegmentPassOutcome {
    /// Nothing changed; keep the current bytes.
    Keep,
    /// Masked replacements were spliced in.
    Rewritten(Vec<u8>),
    /// A segment-moderating member blocked. Carries the firing
    /// guardrail's name and, when the refusal was an availability failure
    /// rather than a content decision, its bounded failure tag — the same
    /// two the `Block` verdict carries, so the `/mcp` tool result says
    /// which of the two happened just like every other family does.
    Block {
        guardrail_name: Option<String>,
        unavailable: Option<String>,
    },
}

/// Segment pass over the request's `params.arguments` string leaves.
async fn moderate_tool_arguments(
    chain: &aisix_guardrails::GuardrailChain,
    body: &[u8],
    monitor_hits: &mut Vec<aisix_core::GuardrailMonitorHit>,
    counts_out: &mut crate::redact::RedactionCounts,
) -> SegmentPassOutcome {
    moderate_selected_segments(
        chain,
        body,
        tool_argument_path,
        true,
        monitor_hits,
        counts_out,
    )
    .await
}

/// Run the chain's segment-moderating members (custom scripts, Bedrock
/// ANONYMIZE) over the string leaves `pred` selects (#1363): collect
/// the decoded slots with the same byte-splice walker the write-back
/// uses, moderate them in ONE chain pass, and splice the positionally
/// aligned masked texts back. Every byte outside a masked span reaches
/// the peer verbatim.
async fn moderate_selected_segments(
    chain: &aisix_guardrails::GuardrailChain,
    body: &[u8],
    pred: fn(&[crate::json_splice::PathSeg]) -> bool,
    input: bool,
    monitor_hits: &mut Vec<aisix_core::GuardrailMonitorHit>,
    counts_out: &mut crate::redact::RedactionCounts,
) -> SegmentPassOutcome {
    // Collect: a rewrite walk that rewrites nothing.
    let mut texts: Vec<String> = Vec::new();
    if let Err(err) = crate::json_splice::rewrite_string_values(body, pred, |t| {
        texts.push(t.to_owned());
        None
    }) {
        // Structurally impossible (every caller's body already parsed as
        // JSON) — fail closed rather than let content bypass the pass.
        tracing::warn!(error = %err, "mcp segment collect walk failed; blocking");
        return SegmentPassOutcome::Block {
            guardrail_name: None,
            unavailable: Some(crate::error::TAG_UNSCANNABLE_BODY.to_owned()),
        };
    }
    // No early return on an empty collect walk — see `redact::moderate_body`.
    // A `tools/call` with `"arguments": {}` has no string leaves, and
    // returning `Keep` here meant a guardrail scoped to the MCP server was
    // never consulted: the tool executed under an unconditional block rule.
    let mut outcome = if input {
        aisix_guardrails::Guardrail::moderate_input_segments(chain, &texts).await
    } else {
        aisix_guardrails::Guardrail::moderate_output_segments(chain, &texts).await
    };
    monitor_hits.append(&mut outcome.monitor_hits);
    if let aisix_guardrails::GuardrailVerdict::Block {
        reason,
        guardrail_name,
        unavailable,
    } = outcome.verdict
    {
        tracing::warn!(
            guardrail_hook = if input { "input" } else { "output" },
            reason = %reason,
            "guardrail blocked MCP content in the segment pass"
        );
        return SegmentPassOutcome::Block {
            guardrail_name,
            unavailable,
        };
    }
    let Some(masked) = outcome.masked else {
        return SegmentPassOutcome::Keep;
    };
    // The chain fold already refuses a member whose rewrite drifts from
    // the offered count, so a drift HERE means the chain itself broke its
    // contract — the same invariant-violation class as a splice failure,
    // and it takes the same fail-closed arm.
    if masked.len() != texts.len() {
        tracing::warn!(
            offered = texts.len(),
            masked = masked.len(),
            "mcp segment mask drifted from the collect walk; blocking"
        );
        return SegmentPassOutcome::Block {
            guardrail_name: None,
            unavailable: Some(crate::error::TAG_MASK_WRITEBACK_FAILED.to_owned()),
        };
    }
    let mut cursor = 0usize;
    match crate::json_splice::rewrite_string_values(body, pred, |t| {
        let i = cursor;
        cursor += 1;
        match masked.get(i) {
            Some(m) if m != t => Some(m.clone()),
            _ => None,
        }
    }) {
        Ok(None) => SegmentPassOutcome::Keep,
        Ok(Some(bytes)) => {
            crate::redact::merge_counts(counts_out, outcome.counts);
            SegmentPassOutcome::Rewritten(bytes)
        }
        Err(err) => {
            tracing::warn!(error = %err, "mcp segment mask splice failed; blocking");
            SegmentPassOutcome::Block {
                guardrail_name: None,
                unavailable: Some(crate::error::TAG_MASK_WRITEBACK_FAILED.to_owned()),
            }
        }
    }
}

/// The post-mask content-capture pair for a tool call: `prompt` is the
/// (rewritten) `params.arguments`, `response` the (rewritten) `result`.
/// Both re-serialise through `Value` — capture is telemetry, not wire
/// bytes, so canonical key order is fine here.
fn tool_call_capture(
    cap: u32,
    request_bytes: Option<&[u8]>,
    result_bytes: Option<&[u8]>,
) -> aisix_obs::CapturedContent {
    let prompt = request_bytes
        .and_then(|b| serde_json::from_slice::<JsonRpcPeek>(b).ok())
        .and_then(|p| p.params)
        .and_then(|p| p.arguments)
        .map(|v| v.to_string())
        .unwrap_or_default();
    let response = result_bytes
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(b).ok())
        .and_then(|mut v| v.get_mut("result").map(serde_json::Value::take))
        .map(|v| v.to_string())
        .unwrap_or_default();
    aisix_obs::CapturedContent::new(&prompt, &response, cap as usize)
}

/// Outcome of the output-hook guardrail pass over an MCP tool result.
enum ToolResultOutcome {
    /// Reject the tool result. Carries the firing guardrail's name (or
    /// `None` for a fail-closed block on an unparseable body / splice
    /// failure) and, when the refusal was an availability failure rather
    /// than a content decision, its bounded failure tag.
    Block {
        guardrail_name: Option<String>,
        unavailable: Option<String>,
    },
    /// Release the tool result; `Some` carries the mask-rewritten body
    /// bytes and the per-detector counts.
    Allow(Option<(Vec<u8>, crate::redact::RedactionCounts)>),
}

/// Run the output guardrail chain over an MCP tool result: verdict AND
/// mask write-back (AISIX-Cloud#1330). The tool result's text is fed to
/// `check_output` as assistant text, the same hook the LLM response path
/// uses; a protocol-level error envelope (no `result`) has nothing to
/// scan and is allowed. When the chain carries a sync redactor, masked
/// spans are spliced back into the raw body bytes — every byte outside a
/// masked span reaches the client verbatim.
async fn apply_output_guardrails(
    chain: &aisix_guardrails::GuardrailChain,
    response_bytes: &[u8],
    tool: &str,
    monitor_hits: &mut Vec<aisix_core::GuardrailMonitorHit>,
) -> ToolResultOutcome {
    // Fail closed on an unparseable body. The `/mcp` gateway is configured
    // `json_response = true`, so a `tools/call` returns a single
    // `application/json` object; a body that does not parse (e.g. if that ever
    // regressed to SSE framing) must not slip an unscanned tool result past the
    // guardrail — block rather than allow.
    //
    // Only when a guardrail would have READ the result and refuses when it
    // cannot evaluate, though. The chain is resolved once for both
    // directions, so it is non-empty for a tool call screened on the request
    // side alone; refusing the RESULT on the strength of an input-hook
    // attachment would refuse a body that attachment was never going to
    // inspect. `output_fail_open: true` opts out on the same grounds the
    // input side honours `fail_open`.
    let value: serde_json::Value = match serde_json::from_slice(response_bytes) {
        Ok(value) => value,
        Err(_) if !aisix_guardrails::Guardrail::refuses_unevaluable_output(chain) => {
            // Released unscanned — a bypass, under the tag the fail-closed
            // arm below refuses with. See the same pair in `messages.rs`.
            chain.record_unevaluable_output_bypass(crate::error::TAG_UNSCANNABLE_BODY);
            return ToolResultOutcome::Allow(None);
        }
        Err(_) => {
            return ToolResultOutcome::Block {
                guardrail_name: None,
                unavailable: Some(crate::error::TAG_UNSCANNABLE_BODY.to_owned()),
            }
        }
    };
    // A protocol-level error envelope (no `result`) has no tool output to scan.
    let Some(result) = value.get("result") else {
        return ToolResultOutcome::Allow(None);
    };
    // Scan the client-visible tool text — the decoded content-block strings,
    // not the serialized JSON envelope. This keeps MCP output and LLM output
    // on the same representation: a keyword guardrail sees the decoded prose,
    // so envelope field names (`content`, `type`, `text`) can't trip a false
    // positive, and escaped characters can't hide blocked content.
    let mut scanned: Vec<String> = Vec::new();
    if let Some(blocks) = result.get("content").and_then(|c| c.as_array()) {
        for block in blocks {
            // The `text` string of any block, plus the block-level
            // `description`/`title` a `resource_link` carries — all string
            // VALUES, never envelope keys, so scanning them is safe and
            // strictly wider than the old `type == "text"` filter. A
            // link's `name`/`uri` are deliberately NOT scanned or masked:
            // they address the resource, and rewriting an identifier
            // breaks the client's follow-up fetch.
            for key in ["text", "description", "title"] {
                if let Some(text) = block.get(key).and_then(|t| t.as_str()) {
                    if !text.is_empty() {
                        scanned.push(text.to_owned());
                    }
                }
            }
            // Embedded resource (`type: "resource"`): the payload rides
            // `resource.text` — the natural shape for a log file — and
            // previously escaped the scan entirely whenever a sibling
            // text block kept `scanned` non-empty (AISIX-Cloud#1330).
            // base64 `blob` resources are NOT decoded here (deliberate:
            // mimeType allowlist + size caps are an open design point).
            if let Some(text) = block
                .get("resource")
                .and_then(|r| r.get("text"))
                .and_then(|t| t.as_str())
            {
                if !text.is_empty() {
                    scanned.push(text.to_owned());
                }
            }
        }
    }
    // `structuredContent` is serialized to the client ALONGSIDE `content`, and
    // the spec only RECOMMENDS mirroring it into a text block — so a tool can
    // return clean prose and carry the sensitive value here. Scan its string
    // leaves, for the same reason the content blocks are decoded first: the
    // values are the tool's data, while the object keys are its output schema.
    if let Some(structured) = result.get("structuredContent") {
        collect_string_leaves(structured, &mut scanned);
    }
    // Fall back to the whole serialized result for non-standard shapes so
    // nothing escapes inspection.
    let result_text = if scanned.is_empty() {
        result.to_string()
    } else {
        scanned.join("\n")
    };
    let resp = aisix_gateway::ChatResponse {
        id: String::new(),
        model: String::new(),
        message: aisix_gateway::ChatMessage::assistant(result_text),
        finish_reason: aisix_gateway::FinishReason::Stop,
        usage: aisix_gateway::UsageStats::new(0, 0),
    };
    // Segment-moderating members answer through the segment pass below.
    let (verdict, hits) =
        aisix_guardrails::Guardrail::check_output_non_segment_observed(chain, &resp).await;
    monitor_hits.extend(hits);
    if let aisix_guardrails::GuardrailVerdict::Block {
        reason,
        guardrail_name,
        unavailable,
    } = verdict
    {
        tracing::warn!(
            guardrail_hook = "output",
            tool = %tool,
            reason = %reason,
            "guardrail blocked MCP tool result"
        );
        return ToolResultOutcome::Block {
            guardrail_name,
            unavailable,
        };
    }
    // Mask write-back over the same surface the scan covers
    // (`tool_result_path`; `name`/`uri` stay untouched — they address a
    // resource; see the scan-loop comment). Two write-back channels
    // compose: the sync per-field redactors (kind=pii mask rules), then
    // the async segment pass (custom scripts, Bedrock ANONYMIZE) over
    // whatever the sync pass produced.
    let mut counts = crate::redact::RedactionCounts::new();
    let mut current: Option<Vec<u8>> = None;
    if aisix_guardrails::Guardrail::redacts_output(chain) {
        let rewritten =
            crate::json_splice::rewrite_string_values(response_bytes, tool_result_path, |text| {
                aisix_guardrails::Guardrail::redact_output_text(chain, text).map(|r| {
                    crate::redact::merge_counts(&mut counts, r.counts);
                    r.text
                })
            });
        match rewritten {
            Ok(None) => {}
            Ok(Some(bytes)) => current = Some(bytes),
            Err(err) => {
                // Structurally impossible (the body parsed above); fail
                // closed rather than release a result the mask policy
                // should rewrite.
                tracing::warn!(
                    tool = %tool,
                    error = %err,
                    "mcp output mask splice failed; blocking tool result",
                );
                return ToolResultOutcome::Block {
                    guardrail_name: None,
                    unavailable: Some(crate::error::TAG_MASK_WRITEBACK_FAILED.to_owned()),
                };
            }
        }
    }
    if aisix_guardrails::Guardrail::moderates_segments(chain) {
        let body_now: &[u8] = current.as_deref().unwrap_or(response_bytes);
        match moderate_selected_segments(
            chain,
            body_now,
            tool_result_path,
            false,
            monitor_hits,
            &mut counts,
        )
        .await
        {
            SegmentPassOutcome::Keep => {}
            SegmentPassOutcome::Rewritten(bytes) => current = Some(bytes),
            SegmentPassOutcome::Block {
                guardrail_name,
                unavailable,
            } => {
                return ToolResultOutcome::Block {
                    guardrail_name,
                    unavailable,
                }
            }
        }
    }
    ToolResultOutcome::Allow(current.map(|bytes| (bytes, counts)))
}

/// Push every non-empty string leaf of `value` — walking objects and arrays —
/// onto `out`, in document order. Object KEYS are skipped: they are the tool's
/// declared output-schema field names rather than its data, so scanning them
/// would reintroduce the field-name false positives that decoding the content
/// blocks avoids. The walk is iterative, so a deeply nested result cannot
/// recurse the handler's stack.
fn collect_string_leaves(value: &serde_json::Value, out: &mut Vec<String>) {
    let mut stack = vec![value];
    while let Some(node) = stack.pop() {
        match node {
            serde_json::Value::String(text) if !text.is_empty() => out.push(text.clone()),
            serde_json::Value::Array(items) => stack.extend(items.iter().rev()),
            serde_json::Value::Object(map) => stack.extend(map.values().rev()),
            _ => {}
        }
    }
}

/// Emit a usage event for a single MCP tool call into the same sink as LLM
/// usage. MCP calls carry no token cost yet, so token/cost fields stay zero;
/// the event records who called which tool, the outcome, and the latency.
#[allow(clippy::too_many_arguments)]
fn emit_tool_call_usage(
    state: &ProxyState,
    // The request's snapshot, loaded once by the handler (#941).
    snap: &aisix_core::AisixSnapshot,
    auth: &AuthenticatedKey,
    request_id: &str,
    mcp_server: &str,
    mcp_tool: &str,
    status_code: u16,
    latency: Duration,
    guardrail_blocked: bool,
    guardrail_monitor_hits: Vec<aisix_core::GuardrailMonitorHit>,
    // Per-detector mask counts from the write-back passes (names only,
    // never values — #932 no-leak).
    redacted_entity_counts: crate::redact::RedactionCounts,
    // The request's resolved chain, or `None` when the request was
    // refused before one existed (the quota path). Source of BOTH
    // guardrail attributions on the event: the `{kind, hook}` set that
    // governed the call, and the enforced hits it produced
    // (AISIX-Cloud#1330) — deriving them here rather than at each call
    // site keeps the five emit legs from drifting apart.
    guardrail_chain: Option<&aisix_guardrails::GuardrailChain>,
    // Post-mask captured args/result for full-content exporters
    // (AISIX-Cloud#1330); `None` when no exporter captures content.
    content: Option<&aisix_obs::CapturedContent>,
    trace: Option<&std::sync::Arc<aisix_obs::RequestTraceBundle>>,
    // Whether the tool call reached its upstream MCP server — false for a
    // quota rejection or an input-guardrail block, which refuse before any
    // upstream contact and must not export an upstream CLIENT span.
    dispatched: bool,
) {
    let mut event = UsageEvent {
        request_id: request_id.to_string(),
        occurred_at: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        api_key_id: auth.entry.id.clone(),
        status_code,
        redacted_entity_counts,
        // Single-attempt endpoint: the attempt spans the whole request, so
        // the upstream figure and what the caller waited for coincide.
        upstream_latency_ms: latency.as_millis().min(u32::MAX as u128) as u32,
        downstream_latency_ms: latency.as_millis().min(u32::MAX as u128) as u32,
        inbound_protocol: "mcp".to_string(),
        mcp_server_name: mcp_server.to_string(),
        mcp_tool_name: mcp_tool.to_string(),
        guardrail_blocked,
        guardrail_monitor_hits,
        applied_guardrails: guardrail_chain
            .map(|c| c.applied().to_vec())
            .unwrap_or_default(),
        guardrail_enforced_hits: guardrail_chain
            .map(|c| c.enforced_hits())
            .unwrap_or_default(),
        guardrail_scores: guardrail_chain.map(|c| c.scores()).unwrap_or_default(),
        guardrail_bypassed_reason: guardrail_chain
            .and_then(|c| c.bypass_reason())
            .unwrap_or_default(),
        ..Default::default()
    };
    crate::usage_attr::apply_caller_identity(
        &mut event,
        auth.jwt.as_ref(),
        auth.key().user_id.as_deref(),
        auth.key().user_name.as_deref(),
    );
    crate::usage_attr::apply_auth_type(&mut event, auth);
    // A tool call resolves neither a model nor a ProviderKey, so the
    // attribution labels are the placeholder — present so this family has
    // ONE label set across every handler (AISIX-Cloud#1317).
    // #698: both emit legs (CP sink + per-env exporter fan-out) go through
    // the shared chokepoint — pre-fix MCP usage reached only the CP sink, so
    // exporters never saw /mcp traffic. `content` carries the POST-MASK
    // tool args/result for full-content exporters (AISIX-Cloud#1330).
    crate::usage_attr::emit_usage(
        state,
        snap,
        crate::operation::MCP,
        event,
        aisix_obs::UsageEventLabels::default(),
        content,
        trace,
        /* terminal */ true,
        dispatched,
    );
}

/// Reject a request whose `MCP-Protocol-Version` header names a version
/// `/mcp` does not serve: HTTP 400 with a JSON-RPC error envelope (echoing
/// the request id when one was parseable), listing the supported versions.
/// Returns `None` when the header is absent (spec backwards-compatibility:
/// treated as `2025-03-26` downstream) or names a supported version.
///
/// The header value is caller-controlled; the echo in the message is
/// control-stripped and length-bounded so a hostile value cannot inject log
/// lines or bloat the response.
fn reject_unsupported_protocol_version(
    headers: &axum::http::HeaderMap,
    peek: Option<&JsonRpcPeek>,
) -> Option<Response> {
    const MAX_ECHO: usize = 64;
    let value = headers.get("mcp-protocol-version")?;
    let message = match value.to_str() {
        Ok(v) if aisix_mcp::SUPPORTED_PROTOCOL_VERSION_NAMES.contains(&v) => return None,
        Ok(v) => {
            let echoed: String = v
                .chars()
                .filter(|c| !c.is_control())
                .take(MAX_ECHO)
                .collect();
            format!("unsupported MCP-Protocol-Version: {echoed}")
        }
        Err(_) => "invalid MCP-Protocol-Version header".to_string(),
    };
    // Echo only a VALID JSON-RPC id (string or number). A malformed request
    // carrying an object/array id would otherwise be reflected into an
    // id-invalid response envelope.
    let id = peek
        .and_then(|p| p.id.clone())
        .filter(|id| id.is_string() || id.is_number())
        .unwrap_or(serde_json::Value::Null);
    let body = serde_json::json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": {
            // JSON-RPC 2.0 "Invalid Request": defined identically across
            // every MCP generation, unlike the 2026-07-28-only -32022.
            "code": -32600,
            "message": message,
            "data": { "supported": aisix_mcp::SUPPORTED_PROTOCOL_VERSION_NAMES },
        }
    });
    Some(
        (
            StatusCode::BAD_REQUEST,
            [(axum::http::header::CONTENT_TYPE, "application/json")],
            body.to_string(),
        )
            .into_response(),
    )
}

/// Build the MCP-native response for a guardrail block: a `tools/call` result
/// flagged `isError`, echoing the request id, served as HTTP 200 with a JSON
/// body (the MCP Streamable HTTP shape). Both the input and output hooks funnel
/// through here; `side` (`"tool call"` for input arguments, `"tool result"` for
/// output) selects the caller-visible wording.
///
/// A tool-execution error rather than a JSON-RPC protocol error: MCP separates
/// "this request was not valid" (a protocol error, which a client surfaces as a
/// transport-level failure) from "the tool call did not succeed" (`isError` on
/// the result, which the calling agent reads as tool output and can adapt to).
/// A policy rejection is the second kind — the request was well-formed and the
/// caller should learn, in-band, that content policy stopped it.
fn jsonrpc_guardrail_block(
    id: Option<serde_json::Value>,
    side: &str,
    guardrail_name: Option<&str>,
    unavailable: Option<&str>,
) -> Response {
    let message = crate::error::guardrail_block_message(side, guardrail_name, unavailable);
    let body = serde_json::json!({
        "jsonrpc": "2.0",
        "id": id.unwrap_or(serde_json::Value::Null),
        "result": {
            "content": [{ "type": "text", "text": message }],
            "isError": true,
        }
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
            request_id: Default::default(),
            url_rewrites: Vec::new(),
            tls: None,
            thread_per_core: None,
            workers: None,
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
            "mcp_access": { "allow": ["*"] },
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
                "mcp_access": { "allow": ["*"] },
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
    async fn empty_key_allow_list_rejects_tool_calls_at_the_acl() {
        // An `mcp_access` block whose allow side is empty leaves the key no
        // MCP access even though an env policy grants everything. The
        // rejection is the ACL's neutral "not available" (reached before
        // upstream routing), not the router's "unknown MCP server", which
        // proves the endpoint resolves the layered ACL rather than falling
        // through to routing.
        let key_hash = ApiKey::hash_bearer(TOKEN);
        let apikey: ApiKey = serde_json::from_value(serde_json::json!({
            "key_hash": key_hash,
            "allowed_models": ["*"],
            "mcp_access": { "allow": [] },
        }))
        .expect("valid apikey");
        let policy: aisix_core::models::McpPolicy = serde_json::from_value(serde_json::json!({
            "scope": "env",
            "allow": ["*"],
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
        let resp = router
            .oneshot(tools_call_request())
            .await
            .expect("router responds");
        let body = body_json(resp).await;
        let message = body["error"]["message"].as_str().unwrap_or_default();
        assert!(
            message.contains("not available"),
            "a key allowing nothing must be rejected by the ACL, got: {body}"
        );
    }

    #[tokio::test]
    async fn unconfigured_key_has_no_mcp_access_at_the_endpoint() {
        // No policy anywhere and no `mcp_access` block: the grant is empty
        // rather than unrestricted, so the call never reaches routing.
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

        let router = router_with(snapshot);
        let resp = router
            .oneshot(tools_call_request())
            .await
            .expect("router responds");
        let body = body_json(resp).await;
        let message = body["error"]["message"].as_str().unwrap_or_default();
        assert!(
            message.contains("not available"),
            "an unconfigured key must have no MCP access, got: {body}"
        );
    }

    #[tokio::test]
    async fn env_policy_deny_overlays_a_wildcard_key_at_the_endpoint() {
        // An env policy allowing everything but denying exactly one tool:
        // the denied tool is rejected by the ACL while any other name still
        // reaches routing (and fails as "unknown MCP server" — no upstreams
        // are registered).
        let key_hash = ApiKey::hash_bearer(TOKEN);
        let apikey: ApiKey = serde_json::from_value(serde_json::json!({
            "key_hash": key_hash,
            "allowed_models": ["*"],
        }))
        .expect("valid apikey");
        let policy: aisix_core::models::McpPolicy = serde_json::from_value(serde_json::json!({
            "scope": "env",
            "allow": ["*"],
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
            "env-policy deny must subtract from the inherited grant, got: {body}"
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
            "a non-denied tool must still pass the ACL, got: {body}"
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

    // ─── Anonymous access (AISIX-Cloud#1313) ─────────────────────────

    /// A snapshot whose environment serves anonymous callers as `ak-1`,
    /// with `anon` merged into the anonymous block.
    fn snapshot_with_anonymous(anon: serde_json::Value) -> AisixSnapshot {
        let snapshot = snapshot_with_key();
        let mut block = serde_json::json!({
            "api_key_id": "ak-1",
            "source_cidrs": ["10.0.0.0/8"],
        });
        let obj = block.as_object_mut().unwrap();
        for (k, v) in anon.as_object().expect("anonymous overrides") {
            obj.insert(k.clone(), v.clone());
        }
        let settings: aisix_core::models::McpAuthSettings =
            serde_json::from_value(serde_json::json!({ "anonymous": block }))
                .expect("valid mcp_auth_settings");
        snapshot
            .mcp_auth_settings
            .insert(ResourceEntry::new("env-1", settings, 1));
        // Both registered: `docs` is the allowlisted entry, `kb` the one
        // that exists but is not offered anonymously — so a refusal for
        // `kb` proves the gate, not a missing row.
        insert_mcp_server(&snapshot, "mcp-docs", "docs", true);
        insert_mcp_server(&snapshot, "mcp-kb", "kb", true);
        snapshot
    }

    /// Attach a client socket, since an in-process request carries none
    /// and the anonymous CIDR gate fails closed without a source IP.
    fn from_ip(mut request: HttpRequest<Body>, ip: &str) -> HttpRequest<Body> {
        request
            .extensions_mut()
            .insert(axum::extract::ConnectInfo(std::net::SocketAddr::new(
                ip.parse().expect("test source ip"),
                40000,
            )));
        request
    }

    #[tokio::test]
    async fn anonymous_scoped_entry_serves_a_request_without_credentials() {
        let router = router_with(snapshot_with_anonymous(
            serde_json::json!({ "servers": ["docs"] }),
        ));
        let response = router
            .oneshot(from_ip(
                scoped_request("docs", None, "initialize", serde_json::json!({})),
                "10.1.2.3",
            ))
            .await
            .expect("router responds");
        assert_eq!(
            response.status(),
            StatusCode::OK,
            "a listed server must serve a caller that presents no credential"
        );
    }

    #[tokio::test]
    async fn anonymous_does_not_reach_an_unlisted_scoped_entry() {
        // `kb` IS registered — it is simply not offered anonymously. It
        // must answer exactly like a server that does not exist at all,
        // so an anonymous prober cannot map the registered set by
        // telling 401 from 404.
        let router = router_with(snapshot_with_anonymous(
            serde_json::json!({ "servers": ["docs"] }),
        ));
        for server in ["kb", "ghost"] {
            let response = router
                .clone()
                .oneshot(from_ip(
                    scoped_request(server, None, "initialize", serde_json::json!({})),
                    "10.1.2.3",
                ))
                .await
                .expect("router responds");
            assert_eq!(
                response.status(),
                StatusCode::UNAUTHORIZED,
                "{server} must be indistinguishable from an unknown server"
            );
        }
    }

    #[tokio::test]
    async fn anonymous_aggregated_entry_is_opt_in() {
        // Listing a server opens `/mcp/docs`, never the aggregated
        // endpoint: that one is the OAuth-discovery entry and carries
        // its own switch.
        let router = router_with(snapshot_with_anonymous(
            serde_json::json!({ "servers": ["docs"] }),
        ));
        let response = router
            .oneshot(from_ip(initialize_request(None), "10.1.2.3"))
            .await
            .expect("router responds");
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);

        let router = router_with(snapshot_with_anonymous(
            serde_json::json!({ "servers": ["docs"], "aggregate_entry": true }),
        ));
        let response = router
            .oneshot(from_ip(initialize_request(None), "10.1.2.3"))
            .await
            .expect("router responds");
        assert_ne!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn a_bad_credential_never_downgrades_to_anonymous() {
        // The whole security argument of the feature: anonymous is the
        // no-credential path. A caller that presents something and gets
        // it wrong must fail, or an attacker could probe with garbage
        // and silently land on the anonymous principal's grant.
        let router = router_with(snapshot_with_anonymous(
            serde_json::json!({ "servers": ["docs"], "aggregate_entry": true }),
        ));
        for bad in ["Bearer sk-not-a-real-key", "Basic dXNlcjpwdw==", "Bearer "] {
            let mut request = from_ip(initialize_request(None), "10.1.2.3");
            request.headers_mut().insert(
                axum::http::header::AUTHORIZATION,
                axum::http::HeaderValue::from_str(bad).unwrap(),
            );
            let response = router
                .clone()
                .oneshot(request)
                .await
                .expect("router responds");
            assert_eq!(
                response.status(),
                StatusCode::UNAUTHORIZED,
                "{bad:?} must be rejected, not served anonymously"
            );
        }
    }

    #[tokio::test]
    async fn anonymous_requires_the_source_to_match_the_allowlist() {
        let router = router_with(snapshot_with_anonymous(
            serde_json::json!({ "servers": ["docs"] }),
        ));
        let response = router
            .oneshot(from_ip(
                scoped_request("docs", None, "initialize", serde_json::json!({})),
                "192.0.2.9",
            ))
            .await
            .expect("router responds");
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn anonymous_fails_closed_without_a_resolvable_source() {
        // No client socket: the CIDR gate is the only thing in front of
        // the principal, so an unresolvable source must not pass it.
        let router = router_with(snapshot_with_anonymous(
            serde_json::json!({ "servers": ["docs"] }),
        ));
        let response = router
            .oneshot(scoped_request(
                "docs",
                None,
                "initialize",
                serde_json::json!({}),
            ))
            .await
            .expect("router responds");
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn disabling_the_block_closes_anonymous_access() {
        let router = router_with(snapshot_with_anonymous(
            serde_json::json!({ "servers": ["docs"], "enabled": false }),
        ));
        let response = router
            .oneshot(from_ip(
                scoped_request("docs", None, "initialize", serde_json::json!({})),
                "10.1.2.3",
            ))
            .await
            .expect("router responds");
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn the_anonymous_principal_keeps_its_lifecycle() {
        // A deleted, disabled or expired principal closes the door
        // rather than opening it — the key is the whole governance
        // handle on anonymous traffic.
        for (label, key) in [
            ("missing", serde_json::json!(null)),
            (
                "disabled",
                serde_json::json!({"key_hash": "h", "allowed_models": ["*"], "disabled": true}),
            ),
            (
                "expired",
                serde_json::json!({
                    "key_hash": "h", "allowed_models": ["*"],
                    "expires_at": "2020-01-01T00:00:00Z"
                }),
            ),
        ] {
            let snapshot = snapshot_with_anonymous(serde_json::json!({ "servers": ["docs"] }));
            snapshot.apikeys.remove("ak-1");
            if let Some(doc) = key.as_object() {
                let apikey: ApiKey =
                    serde_json::from_value(serde_json::Value::Object(doc.clone())).unwrap();
                snapshot
                    .apikeys
                    .insert(ResourceEntry::new("ak-1", apikey, 1));
            }
            let response = router_with(snapshot)
                .oneshot(from_ip(
                    scoped_request("docs", None, "initialize", serde_json::json!({})),
                    "10.1.2.3",
                ))
                .await
                .expect("router responds");
            assert_eq!(
                response.status(),
                StatusCode::UNAUTHORIZED,
                "a {label} anonymous principal must close the entry"
            );
        }
    }

    #[tokio::test]
    async fn a_valid_credential_still_authenticates_where_anonymous_is_open() {
        // Anonymous does not shadow the normal path: a real key keeps
        // its own identity (and its own, wider grant) on the same entry.
        let router = router_with(snapshot_with_anonymous(
            serde_json::json!({ "servers": ["docs"], "aggregate_entry": true }),
        ));
        let response = router
            .oneshot(from_ip(initialize_request(Some(TOKEN)), "192.0.2.9"))
            .await
            .expect("router responds");
        assert_ne!(
            response.status(),
            StatusCode::UNAUTHORIZED,
            "an authenticated caller is not subject to the anonymous CIDR gate"
        );
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
    async fn anonymous_tool_call_is_marked_on_the_usage_event() {
        use aisix_obs::{UsageEvent, UsageSink};

        // The principal is a normal key row, so `api_key_id` alone cannot
        // say whether the caller proved it or inherited it from the
        // anonymous entry — `auth_type` is what keeps the two apart in
        // the usage record.
        let (tx, mut rx) = tokio::sync::mpsc::channel::<UsageEvent>(8);
        let snapshot = snapshot_with_anonymous(serde_json::json!({
            "servers": ["docs"], "aggregate_entry": true
        }));
        let handle = SnapshotHandle::new(snapshot);
        let hub = Arc::new(aisix_gateway::Hub::new());
        let state = ProxyState::new(handle, hub, &cfg())
            .without_cache()
            .with_usage_sink(UsageSink::new(tx));
        let router = build_router(state);

        let mut anonymous = tools_call_request();
        anonymous.headers_mut().remove("authorization");
        let _ = router
            .clone()
            .oneshot(from_ip(anonymous, "10.1.2.3"))
            .await
            .expect("router responds");
        let event = rx.try_recv().expect("anonymous tool call is recorded");
        assert_eq!(event.api_key_id, "ak-1");
        assert_eq!(event.auth_type, "anonymous");

        // The same principal reached with its own credential is not.
        let _ = router
            .oneshot(tools_call_request())
            .await
            .expect("router responds");
        let event = rx.try_recv().expect("authenticated tool call is recorded");
        assert_eq!(event.api_key_id, "ak-1");
        assert_eq!(
            event.auth_type, "",
            "the ordinary credential path leaves the field off the wire"
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
        seed_guardrail_with_attachment(
            handle,
            guardrail_json,
            r#"{"guardrail_id":"g1","scope_type":"env","priority":50}"#,
        );
    }

    /// Register an MCP server under `name` with the resource id `id`, so a
    /// tool call naming it resolves the id an `mcp_server`-scoped attachment
    /// matches on.
    fn seed_mcp_server(handle: &SnapshotHandle<AisixSnapshot>, id: &'static str, name: &str) {
        use aisix_core::models::McpServer;
        let server: McpServer = serde_json::from_value(serde_json::json!({
            "name": name,
            // Never dialled: the guardrail verdicts under test are decided
            // before the gateway is built (input) or on a synthesized body
            // (output).
            "url": "http://127.0.0.1:1/mcp",
        }))
        .expect("valid mcp server");
        handle.rcu(|snap| {
            let new = snap.clone();
            new.mcp_servers
                .insert(ResourceEntry::new(id, server.clone(), 1));
            new
        });
    }

    fn seed_guardrail_with_attachment(
        handle: &SnapshotHandle<AisixSnapshot>,
        guardrail_json: &str,
        attachment_json: &str,
    ) {
        use aisix_core::models::{Guardrail, GuardrailAttachment};
        let guardrail: Guardrail = serde_json::from_str(guardrail_json).unwrap();
        let attachment: GuardrailAttachment = serde_json::from_str(attachment_json).unwrap();
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

    /// Verdict-only view over [`apply_output_guardrails`] for the block
    /// tests: `Some(name)` = blocked, `None` = allowed (rewritten or not).
    async fn output_guardrail_block(
        chain: &aisix_guardrails::GuardrailChain,
        response_bytes: &[u8],
        tool: &str,
        monitor_hits: &mut Vec<aisix_core::GuardrailMonitorHit>,
    ) -> Option<Option<String>> {
        match apply_output_guardrails(chain, response_bytes, tool, monitor_hits).await {
            ToolResultOutcome::Block { guardrail_name, .. } => Some(guardrail_name),
            ToolResultOutcome::Allow(_) => None,
        }
    }

    /// Mask-action pii guardrail with a capture-group custom pattern +
    /// literal replacement (AISIX-Cloud#1334), for the write-back tests.
    fn pii_mask_guard(hook: &str) -> String {
        format!(
            r#"{{"name":"pii-mask","kind":"pii","hook_point":"{hook}","custom_patterns":[{{"name":"eda_version","regex":"version\\s*:\\s*(\\d+(?:\\.\\d+)+)","action":"mask","replacement":"***"}}]}}"#,
        )
    }

    fn env_chain_with(guardrail_json: &str) -> aisix_guardrails::GuardrailChain {
        use aisix_guardrails::{LiveGuardrailIndex, RequestContext};
        let handle = SnapshotHandle::new(snapshot_with_key());
        seed_guardrail(&handle, guardrail_json);
        LiveGuardrailIndex::new(handle, None).resolve(&RequestContext {
            passthrough_route_id: "",
            model_id: "",
            mcp_server_id: "",
            api_key_id: "ak-1",
            team_id: None,
        })
    }

    /// AISIX-Cloud#1330: the output hook rewrites masked spans IN PLACE —
    /// content text, embedded resource text, structuredContent leaves —
    /// and every byte outside the hits (key order, whitespace, number
    /// spellings, the envelope) survives verbatim.
    #[tokio::test]
    async fn output_mask_rewrites_in_place_byte_identical_elsewhere() {
        let chain = env_chain_with(&pii_mask_guard("output"));
        let body = concat!(
            r#"{"jsonrpc":"2.0","id":1,"result":{"content":["#,
            r#"{"type":"text","text":"tool version: 12.1 ok"},"#,
            r#"{"type":"resource","resource":{"uri":"file:///run.log","mimeType":"text/plain","text":"Compile version: 2022.4 Elapsed: 12.345s"}}"#,
            r#"], "structuredContent":{"log":"cfg version: 9.0 end","cells": 1e3}}}"#,
        );
        let out = match apply_output_guardrails(&chain, body.as_bytes(), "report", &mut Vec::new())
            .await
        {
            ToolResultOutcome::Allow(Some((bytes, counts))) => {
                assert_eq!(counts.get("eda_version"), Some(&3));
                String::from_utf8(bytes).unwrap()
            }
            other => panic!(
                "expected a rewritten Allow, got {}",
                match other {
                    ToolResultOutcome::Block { .. } => "Block",
                    ToolResultOutcome::Allow(None) => "Allow(None)",
                    ToolResultOutcome::Allow(_) => unreachable!(),
                }
            ),
        };
        assert_eq!(
            out,
            concat!(
                r#"{"jsonrpc":"2.0","id":1,"result":{"content":["#,
                r#"{"type":"text","text":"tool version: *** ok"},"#,
                r#"{"type":"resource","resource":{"uri":"file:///run.log","mimeType":"text/plain","text":"Compile version: *** Elapsed: 12.345s"}}"#,
                r#"], "structuredContent":{"log":"cfg version: *** end","cells": 1e3}}}"#,
            ),
        );
        // Structural acceptance: the rewritten body still parses.
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(
            v["result"]["content"][1]["resource"]["mimeType"],
            "text/plain"
        );

        // A no-hit result is returned with NO rewrite at all.
        let clean = br#"{"jsonrpc":"2.0","id":1,"result":{"content":[{"type":"text","text":"Elapsed: 12.345s"}]}}"#;
        assert!(matches!(
            apply_output_guardrails(&chain, clean, "report", &mut Vec::new()).await,
            ToolResultOutcome::Allow(None),
        ));
    }

    /// The input hook rewrites only the string leaves under
    /// `params.arguments`; the method, tool name, id, and every other
    /// byte are untouched.
    #[test]
    fn input_mask_rewrites_only_arguments() {
        let chain = env_chain_with(&pii_mask_guard("input"));
        let body = r#"{"jsonrpc":"2.0","id":9,"method":"tools/call","params":{"name":"eda__echo","arguments":{"text":"build version: 12.1 done","note":"version untouched"}}}"#;
        let (bytes, counts) = rewrite_tool_arguments(&chain, body.as_bytes())
            .unwrap()
            .expect("a hit rewrites");
        assert_eq!(counts.get("eda_version"), Some(&1));
        assert_eq!(
            String::from_utf8(bytes).unwrap(),
            r#"{"jsonrpc":"2.0","id":9,"method":"tools/call","params":{"name":"eda__echo","arguments":{"text":"build version: *** done","note":"version untouched"}}}"#,
        );
        // No hit → no allocation, original bytes forwarded.
        let clean = br#"{"jsonrpc":"2.0","id":9,"method":"tools/call","params":{"name":"eda__echo","arguments":{"text":"Elapsed: 12.345s"}}}"#;
        assert!(rewrite_tool_arguments(&chain, clean).unwrap().is_none());
    }

    /// AISIX-Cloud#1330 audit chain: a mask that actually fired names the
    /// guardrail ROW that fired it, on the right hook, with the detector
    /// counts. Before this, an enforcing mask left the row name nowhere —
    /// `redacted_entity_counts` carries detector names, not the policy's,
    /// so an operator could see THAT something was masked but never WHICH
    /// policy did it.
    #[test]
    fn enforced_input_mask_names_the_guardrail_row() {
        let chain = env_chain_with(&pii_mask_guard("input"));
        assert!(chain.enforced_hits().is_empty(), "nothing has run yet");

        let body = r#"{"jsonrpc":"2.0","id":9,"method":"tools/call","params":{"name":"eda__echo","arguments":{"text":"build version: 12.1 done"}}}"#;
        rewrite_tool_arguments(&chain, body.as_bytes())
            .unwrap()
            .expect("a hit rewrites");

        let hits = chain.enforced_hits();
        assert_eq!(hits.len(), 1, "one row fired: {hits:?}");
        assert_eq!(hits[0].guardrail_name, "pii-mask");
        assert_eq!(hits[0].hook, "input");
        assert_eq!(hits[0].action, "masked");
        assert_eq!(hits[0].counts.get("eda_version"), Some(&1));
    }

    /// A clean request records nothing: the audit array is evidence that
    /// enforcement happened, so an empty one must mean it did not.
    #[test]
    fn a_clean_request_records_no_enforced_hit() {
        let chain = env_chain_with(&pii_mask_guard("input"));
        let clean = br#"{"jsonrpc":"2.0","id":9,"method":"tools/call","params":{"name":"eda__echo","arguments":{"text":"Elapsed: 12.345s"}}}"#;
        assert!(rewrite_tool_arguments(&chain, clean).unwrap().is_none());
        assert!(chain.enforced_hits().is_empty());
    }

    /// The output hook attributes to the same row, and the many per-leaf
    /// redact calls one tool result triggers coalesce into ONE entry whose
    /// counts are summed — the array is bounded by the chain's member
    /// count, not by how many strings the tool returned.
    #[tokio::test]
    async fn enforced_output_mask_coalesces_per_leaf_calls_into_one_entry() {
        let chain = env_chain_with(&pii_mask_guard("output"));
        let body = concat!(
            r#"{"jsonrpc":"2.0","id":1,"result":{"content":["#,
            r#"{"type":"text","text":"tool version: 12.1 ok"},"#,
            r#"{"type":"resource","resource":{"uri":"file:///run.log","mimeType":"text/plain","text":"Compile version: 2022.4 done"}}"#,
            r#"], "structuredContent":{"log":"cfg version: 9.0 end"}}}"#,
        );
        let outcome =
            apply_output_guardrails(&chain, body.as_bytes(), "report", &mut Vec::new()).await;
        assert!(matches!(outcome, ToolResultOutcome::Allow(Some(_))));

        let hits = chain.enforced_hits();
        assert_eq!(hits.len(), 1, "three masked leaves, one entry: {hits:?}");
        assert_eq!(hits[0].guardrail_name, "pii-mask");
        assert_eq!(hits[0].hook, "output");
        assert_eq!(hits[0].action, "masked");
        assert_eq!(hits[0].counts.get("eda_version"), Some(&3));
    }

    /// An enforcing BLOCK names its row too, with no counts — the block
    /// short-circuits before anything is rewritten, and the operator-facing
    /// reason deliberately stays off the event (#153 no-leak).
    #[tokio::test]
    async fn enforced_output_block_names_the_guardrail_row_without_counts() {
        let chain = env_chain_with(OUTPUT_GUARD);
        let body = br#"{"jsonrpc":"2.0","id":1,"result":{"content":[{"type":"text","text":"forbidden-token"}]}}"#;
        assert!(matches!(
            apply_output_guardrails(&chain, body, "report", &mut Vec::new()).await,
            ToolResultOutcome::Block {
                guardrail_name: Some(_),
                ..
            },
        ));

        let hits = chain.enforced_hits();
        assert_eq!(hits.len(), 1, "{hits:?}");
        assert_eq!(hits[0].guardrail_name, "mcp-output-guard");
        assert_eq!(hits[0].hook, "output");
        assert_eq!(hits[0].action, "blocked");
        assert!(hits[0].counts.is_empty());
    }

    /// The end of the chain: an enforced block on a real `/mcp` request
    /// reaches the emitted usage event, naming the row and the action.
    /// `applied_guardrails` rides the same event — a row that says "masked
    /// by X" while claiming no guardrail governed the call reads as a bug.
    #[tokio::test]
    async fn usage_event_carries_the_enforced_hit_and_the_applied_set() {
        use aisix_obs::{UsageEvent, UsageSink};

        let (tx, mut rx) = tokio::sync::mpsc::channel::<UsageEvent>(8);
        let handle = SnapshotHandle::new(snapshot_with_key());
        let hub = Arc::new(aisix_gateway::Hub::new());
        let state = ProxyState::new(handle.clone(), hub, &cfg())
            .without_cache()
            .with_usage_sink(UsageSink::new(tx));
        let router = build_router(state);
        seed_guardrail(&handle, INPUT_GUARD);

        let _ = router
            .oneshot(tools_call_with_args(
                serde_json::json!({ "q": "forbidden-token" }),
            ))
            .await
            .expect("router responds");

        let event = rx.try_recv().expect("usage event for the blocked call");
        assert!(event.guardrail_blocked);
        assert_eq!(event.guardrail_enforced_hits.len(), 1);
        assert_eq!(
            event.guardrail_enforced_hits[0].guardrail_name,
            "mcp-input-guard"
        );
        assert_eq!(event.guardrail_enforced_hits[0].hook, "input");
        assert_eq!(event.guardrail_enforced_hits[0].action, "blocked");
        assert_eq!(
            event
                .applied_guardrails
                .iter()
                .map(|a| (a.kind.as_str(), a.hook.as_str()))
                .collect::<Vec<_>>(),
            // `hook` is the row's configured `hook_point`, which this
            // guardrail leaves at its default — not the hook it fired on.
            vec![("keyword", "both")],
        );
        // The matched token is the caller's content, not audit data.
        let wire = serde_json::to_string(&event).expect("event serialises");
        assert!(!wire.contains("forbidden-token"), "{wire}");
    }

    /// AISIX-Cloud#1330 scan-surface fix: a forbidden token that appears
    /// ONLY in an embedded resource's text — next to a clean text block —
    /// must block. Pre-fix the `type == "text"` filter dropped the
    /// resource and the non-empty scan set suppressed the fallback, so
    /// exactly this shape (text summary + resource log) went unread.
    #[tokio::test]
    async fn output_guardrail_scans_embedded_resource_text() {
        let chain = env_chain_with(OUTPUT_GUARD);
        let body = br#"{"jsonrpc":"2.0","id":1,"result":{"content":[{"type":"text","text":"summary ok"},{"type":"resource","resource":{"uri":"file:///run.log","text":"log carries forbidden-token here"}}]}}"#;
        assert!(
            output_guardrail_block(&chain, body, "report", &mut Vec::new())
                .await
                .is_some(),
            "resource.text must be scanned even when a text block exists"
        );
    }

    /// Same silent class one sibling over (#1008 audit): a
    /// `resource_link` block carries its data in block-level
    /// `description`/`title`. Both must scan (block) and mask (rewrite);
    /// `name`/`uri` address the resource and stay untouched.
    #[tokio::test]
    async fn output_guardrail_covers_resource_link_description_and_title() {
        // Block rule anchored only in the link description, with a clean
        // sibling text block suppressing the fallback.
        let chain = env_chain_with(OUTPUT_GUARD);
        let body = br#"{"jsonrpc":"2.0","id":1,"result":{"content":[{"type":"text","text":"summary ok"},{"type":"resource_link","uri":"file:///a.log","name":"a.log","description":"holds forbidden-token data"}]}}"#;
        assert!(
            output_guardrail_block(&chain, body, "list", &mut Vec::new())
                .await
                .is_some(),
            "resource_link description must be scanned"
        );

        // Mask rule: description and title rewrite in place; name/uri and
        // every other byte survive verbatim.
        let chain = env_chain_with(&pii_mask_guard("output"));
        let body = concat!(
            r#"{"jsonrpc":"2.0","id":1,"result":{"content":["#,
            r#"{"type":"resource_link","uri":"file:///v.log","name":"run version: 9.9.log","title":"run version: 3.4","description":"log for version: 12.1"}"#,
            r#"]}}"#,
        );
        match apply_output_guardrails(&chain, body.as_bytes(), "list", &mut Vec::new()).await {
            ToolResultOutcome::Allow(Some((bytes, counts))) => {
                assert_eq!(counts.get("eda_version"), Some(&2));
                assert_eq!(
                    String::from_utf8(bytes).unwrap(),
                    concat!(
                        r#"{"jsonrpc":"2.0","id":1,"result":{"content":["#,
                        r#"{"type":"resource_link","uri":"file:///v.log","name":"run version: 9.9.log","title":"run version: ***","description":"log for version: ***"}"#,
                        r#"]}}"#,
                    ),
                    "description/title masked; uri/name untouched",
                );
            }
            _ => panic!("expected a rewritten Allow"),
        }
    }

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
        // guardrail chain LLM input uses — surfaced as an MCP-native tool-error
        // result (HTTP 200) before the gateway/upstream is reached. A
        // distinctive request id (7) proves the handler echoes the caller's id
        // (not a constant) through the block envelope both hooks funnel through.
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
        assert_eq!(
            envelope["result"]["isError"],
            serde_json::json!(true),
            "a policy rejection is a tool-execution error, not a protocol error"
        );
        assert!(
            envelope.get("error").is_none(),
            "a guardrail block must not surface as a JSON-RPC protocol error"
        );
        assert!(
            envelope["result"]["content"][0]["text"]
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
            passthrough_route_id: "",
            model_id: "",
            mcp_server_id: "",
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

    /// The mirror of the `/v1/messages` gate on the response side: `/mcp`
    /// resolves ONE chain for both directions, so a tool call screened on
    /// the request side alone still arrives here with a non-empty chain.
    /// An unparseable tool result must not be refused on the strength of a
    /// row that only ever reads the arguments. Fails on `a456ab71` — the
    /// parse arm blocked unconditionally there.
    #[tokio::test]
    async fn unparseable_tool_result_with_an_input_only_chain_is_allowed() {
        let sse_body = b"event: message\ndata: {\"jsonrpc\":\"2.0\",\"result\":{}}\n\n";

        // Explicitly input-only: `INPUT_GUARD` omits `hook_point`, whose
        // default is `both`, so it would read the result too.
        const INPUT_ONLY_GUARD: &str = r#"{"name":"mcp-input-only","kind":"keyword","hook_point":"input","patterns":[{"kind":"literal","value":"forbidden-token"}]}"#;
        let input_only = env_chain_with(INPUT_ONLY_GUARD);
        assert!(!input_only.is_empty(), "the chain must be non-empty");
        assert!(
            output_guardrail_block(&input_only, sse_body, "echo", &mut Vec::new())
                .await
                .is_none(),
            "an input-hook row never reads the tool result, so it cannot refuse it"
        );

        // An output row that asked to fail open is likewise not a reason to
        // refuse. `kind: keyword` has no `output_fail_open` of its own, so
        // the row-level `fail_open` governs both of its hooks.
        const OUTPUT_OPEN_GUARD: &str = r#"{"name":"mcp-output-open","kind":"keyword","hook_point":"output","fail_open":true,"patterns":[{"kind":"literal","value":"forbidden-token"}]}"#;
        let output_open = env_chain_with(OUTPUT_OPEN_GUARD);
        assert!(!output_open.is_empty(), "the chain must be non-empty");
        assert!(
            output_guardrail_block(&output_open, sse_body, "echo", &mut Vec::new())
                .await
                .is_none(),
            "a row that asked to fail open must not refuse an unparseable result"
        );

        // The output-hook row still fails closed on the same bytes.
        let output = env_chain_with(OUTPUT_GUARD);
        assert!(
            output_guardrail_block(&output, sse_body, "echo", &mut Vec::new())
                .await
                .is_some(),
            "an output-hook row must still fail closed"
        );
    }

    /// Releasing an unparseable tool result under a fail-open OUTPUT row is
    /// a bypass — the client got tool output nothing screened — and it is
    /// recorded under the same tag the fail-closed direction refuses with.
    ///
    /// The input-only chain is the control: nothing in it was ever going to
    /// read the result, so releasing it bypassed nothing and the field must
    /// stay empty. Tagging that case would make the field fire on results
    /// that were never going to be screened.
    #[tokio::test]
    async fn releasing_an_unparseable_tool_result_records_a_bypass_only_when_a_row_read_it() {
        let sse_body = b"event: message\ndata: {\"jsonrpc\":\"2.0\",\"result\":{}}\n\n";

        const OUTPUT_OPEN_GUARD: &str = r#"{"name":"mcp-output-open","kind":"keyword","hook_point":"output","fail_open":true,"patterns":[{"kind":"literal","value":"forbidden-token"}]}"#;
        let output_open = env_chain_with(OUTPUT_OPEN_GUARD);
        assert!(
            output_guardrail_block(&output_open, sse_body, "echo", &mut Vec::new())
                .await
                .is_none(),
            "premise: a fail-open output row must release the result"
        );
        assert_eq!(
            output_open.bypass_reason().as_deref(),
            Some(crate::error::TAG_UNSCANNABLE_BODY),
        );

        const INPUT_ONLY_GUARD: &str = r#"{"name":"mcp-input-only","kind":"keyword","hook_point":"input","patterns":[{"kind":"literal","value":"forbidden-token"}]}"#;
        let input_only = env_chain_with(INPUT_ONLY_GUARD);
        assert!(
            output_guardrail_block(&input_only, sse_body, "echo", &mut Vec::new())
                .await
                .is_none(),
            "premise: an input-only row must release the result"
        );
        assert_eq!(
            input_only.bypass_reason(),
            None,
            "nothing in this chain reads the result, so nothing was bypassed",
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
            passthrough_route_id: "",
            model_id: "",
            mcp_server_id: "",
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
        // the result shape/status/content-type is caught without an rmcp
        // upstream.
        let resp = jsonrpc_guardrail_block(
            Some(serde_json::json!(42)),
            "tool result",
            Some("mcp-output-guard"),
            None,
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
        // A tool-execution error (`isError` on the result), not a JSON-RPC
        // protocol error: the calling agent reads the rejection as tool output
        // it can react to instead of a transport-level failure.
        assert_eq!(v["result"]["isError"], serde_json::json!(true));
        assert!(
            v.get("error").is_none(),
            "a block envelope carries no protocol error"
        );
        assert_eq!(v["result"]["content"][0]["type"], "text");
        assert!(
            v["result"]["content"][0]["text"]
                .as_str()
                .unwrap_or_default()
                .contains("tool result blocked by content policy"),
            "expected the output-side wording, got: {v}"
        );
    }

    /// A tool result can carry data in `structuredContent` that is NOT mirrored
    /// into a text content block (the spec only recommends mirroring), and the
    /// gateway relays that field to the client verbatim. Scanning only the
    /// content blocks would therefore let it through unread.
    #[tokio::test]
    async fn output_guardrail_scans_structured_content() {
        use aisix_guardrails::{LiveGuardrailIndex, RequestContext};

        let handle = SnapshotHandle::new(snapshot_with_key());
        seed_guardrail(&handle, OUTPUT_GUARD);
        let chain = LiveGuardrailIndex::new(handle, None).resolve(&RequestContext {
            passthrough_route_id: "",
            model_id: "",
            mcp_server_id: "",
            api_key_id: "ak-1",
            team_id: None,
        });

        // Clean prose, sensitive structured payload — the client sees both.
        let structured_only = br#"{"jsonrpc":"2.0","id":1,"result":{"content":[{"type":"text","text":"lookup ok"}],"structuredContent":{"record":{"notes":["forbidden-token"]}}}}"#;
        assert!(
            output_guardrail_block(&chain, structured_only, "lookup", &mut Vec::new())
                .await
                .is_some(),
            "structuredContent reaches the client, so it must be scanned"
        );

        // A result with no content blocks at all — a tool may return only
        // structured output — is scanned through the same path.
        let no_content = br#"{"jsonrpc":"2.0","id":1,"result":{"content":[],"structuredContent":{"note":"forbidden-token"}}}"#;
        assert!(
            output_guardrail_block(&chain, no_content, "lookup", &mut Vec::new())
                .await
                .is_some(),
            "a structured-only result must still be scanned"
        );

        // Clean on both sides passes.
        let clean = br#"{"jsonrpc":"2.0","id":1,"result":{"content":[{"type":"text","text":"lookup ok"}],"structuredContent":{"record":{"notes":["all good"]}}}}"#;
        assert!(
            output_guardrail_block(&chain, clean, "lookup", &mut Vec::new())
                .await
                .is_none(),
            "a clean structured result must not be blocked"
        );
    }

    /// The structured walk carries the same "scan data, not field names" rule
    /// the content blocks follow: an object KEY matching the pattern is the
    /// tool's output schema, not its data, and must not fire.
    #[tokio::test]
    async fn structured_content_keys_do_not_trip_the_guardrail() {
        use aisix_guardrails::{LiveGuardrailIndex, RequestContext};

        const KEY_NAME_GUARD: &str = r#"{"name":"key-name-guard","kind":"keyword","hook_point":"output","patterns":[{"kind":"literal","value":"ssn"}]}"#;
        let handle = SnapshotHandle::new(snapshot_with_key());
        seed_guardrail(&handle, KEY_NAME_GUARD);
        let chain = LiveGuardrailIndex::new(handle, None).resolve(&RequestContext {
            passthrough_route_id: "",
            model_id: "",
            mcp_server_id: "",
            api_key_id: "ak-1",
            team_id: None,
        });

        let key_only = br#"{"jsonrpc":"2.0","id":1,"result":{"content":[],"structuredContent":{"ssn":"redacted upstream"}}}"#;
        assert!(
            output_guardrail_block(&chain, key_only, "lookup", &mut Vec::new())
                .await
                .is_none(),
            "a schema field name must not be treated as tool data"
        );

        // The same pattern in a VALUE fires, proving the guardrail is live here.
        let value_hit = br#"{"jsonrpc":"2.0","id":1,"result":{"content":[],"structuredContent":{"field":"ssn 123-45-6789"}}}"#;
        assert!(
            output_guardrail_block(&chain, value_hit, "lookup", &mut Vec::new())
                .await
                .is_some(),
            "the pattern in a structured VALUE must block"
        );
    }

    /// An `mcp_server`-scoped attachment governs only the tool calls routed to
    /// that server — the dimension a Model scope cannot express for MCP.
    #[tokio::test]
    async fn mcp_server_scoped_guardrail_applies_only_to_that_server() {
        let handle = SnapshotHandle::new(snapshot_with_key());
        let hub = Arc::new(aisix_gateway::Hub::new());
        let state = ProxyState::new(handle.clone(), hub, &cfg()).without_cache();
        let router = build_router(state);
        seed_mcp_server(&handle, "mcp-ghost", "ghost");
        seed_mcp_server(&handle, "mcp-other", "other");
        seed_guardrail_with_attachment(
            &handle,
            INPUT_GUARD,
            r#"{"guardrail_id":"g1","scope_type":"mcp_server","scope_id":"mcp-ghost","priority":50}"#,
        );

        let blocked_body = |name: &str| {
            mcp_request(
                "tools/call",
                serde_json::json!({ "name": name, "arguments": { "q": "forbidden-token" } }),
            )
        };
        let body_text = |resp: Response| async move {
            let bytes = axum::body::to_bytes(resp.into_body(), 64 * 1024)
                .await
                .expect("read body");
            String::from_utf8_lossy(&bytes).into_owned()
        };

        let scoped = router
            .clone()
            .oneshot(blocked_body("ghost__tool"))
            .await
            .expect("router responds");
        assert!(
            body_text(scoped).await.contains("content policy"),
            "the attached server's tool call must be blocked"
        );

        let other = router
            .oneshot(blocked_body("other__tool"))
            .await
            .expect("router responds");
        assert!(
            !body_text(other).await.contains("content policy"),
            "a server the guardrail is not attached to must be untouched"
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
            "mcp_access": { "allow": ["*"] },
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

    /// A `/mcp` request for `method` carrying an explicit
    /// `MCP-Protocol-Version` header (and matching request `_meta` for the
    /// stateless `2026-07-28` shape when `modern` is set).
    fn versioned_request(
        version: &str,
        method: &str,
        params: serde_json::Value,
    ) -> HttpRequest<Body> {
        let body = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 7,
            "method": method,
            "params": params
        });
        HttpRequest::post("/mcp")
            .header("host", "mcp.aisix.example.com")
            .header("content-type", "application/json")
            .header("accept", "application/json, text/event-stream")
            .header("authorization", format!("Bearer {TOKEN}"))
            .header("mcp-protocol-version", version)
            .header("mcp-method", method)
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    /// A version outside the served set — including `2024-11-05`, which
    /// rmcp's own transport check would ADMIT (it is in KNOWN_VERSIONS) —
    /// is rejected by the proxy gate: HTTP 400, JSON-RPC envelope, the
    /// request id echoed, the supported list attached
    /// (AISIX-Cloud#1148 / #1144).
    #[tokio::test]
    async fn unsupported_protocol_version_header_is_rejected_in_the_envelope() {
        let router = router_with(snapshot_with_key());
        for version in ["2024-11-05", "not-a-version"] {
            let response = router
                .clone()
                .oneshot(versioned_request(
                    version,
                    "tools/list",
                    serde_json::json!({}),
                ))
                .await
                .expect("router responds");
            assert_eq!(
                response.status(),
                StatusCode::BAD_REQUEST,
                "{version} must be rejected"
            );
            let content_type = response
                .headers()
                .get("content-type")
                .and_then(|v| v.to_str().ok())
                .unwrap_or_default()
                .to_string();
            assert!(
                content_type.starts_with("application/json"),
                "the rejection must use the JSON envelope, not text/plain \
                 (got {content_type})"
            );
            let bytes = to_bytes(response.into_body(), 1_048_576)
                .await
                .expect("read body");
            let body: serde_json::Value = serde_json::from_slice(&bytes).expect("json body");
            assert_eq!(body["jsonrpc"], "2.0", "{body}");
            assert_eq!(body["id"], 7, "request id must be echoed: {body}");
            assert_eq!(body["error"]["code"], -32600, "{body}");
            let supported: Vec<&str> = body["error"]["data"]["supported"]
                .as_array()
                .unwrap_or_else(|| panic!("supported list missing: {body}"))
                .iter()
                .filter_map(|v| v.as_str())
                .collect();
            assert_eq!(
                supported,
                aisix_mcp::SUPPORTED_PROTOCOL_VERSION_NAMES.to_vec()
            );
        }
    }

    /// Hostile header values cannot abuse the rejection envelope: an
    /// overlong value is truncated to 64 echoed characters, a non-ASCII
    /// (obs-text) value gets the static message (never echoed), and an
    /// object request id — invalid JSON-RPC — is sanitized to `null`
    /// rather than reflected.
    #[tokio::test]
    async fn version_gate_rejection_envelope_is_hardened() {
        let router = router_with(snapshot_with_key());

        // Overlong: 300 chars in, at most 64 echoed back out.
        let long_version = "v".repeat(300);
        let response = router
            .clone()
            .oneshot(versioned_request(
                &long_version,
                "tools/list",
                serde_json::json!({}),
            ))
            .await
            .expect("router responds");
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let bytes = to_bytes(response.into_body(), 1_048_576)
            .await
            .expect("read body");
        let body: serde_json::Value = serde_json::from_slice(&bytes).expect("json body");
        let message = body["error"]["message"].as_str().unwrap_or_default();
        let echoed = message
            .strip_prefix("unsupported MCP-Protocol-Version: ")
            .unwrap_or_else(|| panic!("unexpected message shape: {message}"));
        assert_eq!(echoed.chars().count(), 64, "echo must be truncated");

        // Non-ASCII (obs-text) header bytes: `to_str()` fails, the static
        // message is used, nothing of the value is reflected.
        let request = HttpRequest::post("/mcp")
            .header("host", "mcp.aisix.example.com")
            .header("content-type", "application/json")
            .header("accept", "application/json, text/event-stream")
            .header("authorization", format!("Bearer {TOKEN}"))
            .header(
                "mcp-protocol-version",
                axum::http::HeaderValue::from_bytes(&[0x32, 0x30, 0x32, 0x35, 0x80, 0xff])
                    .expect("obs-text bytes are valid header bytes"),
            )
            .body(Body::from(
                serde_json::json!({
                    "jsonrpc": "2.0", "id": 9, "method": "tools/list", "params": {}
                })
                .to_string(),
            ))
            .unwrap();
        let response = router
            .clone()
            .oneshot(request)
            .await
            .expect("router responds");
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let bytes = to_bytes(response.into_body(), 1_048_576)
            .await
            .expect("read body");
        let body: serde_json::Value = serde_json::from_slice(&bytes).expect("json body");
        assert_eq!(
            body["error"]["message"], "invalid MCP-Protocol-Version header",
            "non-UTF8 values must never be echoed: {body}"
        );

        // Object id: invalid per JSON-RPC — sanitized to null, not echoed.
        let body = serde_json::json!({
            "jsonrpc": "2.0",
            "id": { "not": "a valid id" },
            "method": "tools/list",
            "params": {}
        });
        let request = HttpRequest::post("/mcp")
            .header("host", "mcp.aisix.example.com")
            .header("content-type", "application/json")
            .header("accept", "application/json, text/event-stream")
            .header("authorization", format!("Bearer {TOKEN}"))
            .header("mcp-protocol-version", "2024-11-05")
            .body(Body::from(body.to_string()))
            .unwrap();
        let response = router_with(snapshot_with_key())
            .oneshot(request)
            .await
            .expect("router responds");
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let bytes = to_bytes(response.into_body(), 1_048_576)
            .await
            .expect("read body");
        let body: serde_json::Value = serde_json::from_slice(&bytes).expect("json body");
        assert_eq!(
            body["id"],
            serde_json::Value::Null,
            "an object id is not a valid JSON-RPC id and must not be reflected: {body}"
        );
    }

    /// Every version the endpoint serves passes the gate end-to-end: the
    /// legacy generations as plain stateless requests, `2026-07-28` with its
    /// required per-request metadata.
    #[tokio::test]
    async fn supported_protocol_version_headers_pass_the_gate() {
        let router = router_with(snapshot_with_key());
        for version in ["2025-03-26", "2025-06-18", "2025-11-25"] {
            let response = router
                .clone()
                .oneshot(versioned_request(
                    version,
                    "tools/list",
                    serde_json::json!({}),
                ))
                .await
                .expect("router responds");
            assert_eq!(
                response.status(),
                StatusCode::OK,
                "legacy {version} tools/list must be served"
            );
        }
        let modern = router
            .oneshot(versioned_request(
                "2026-07-28",
                "tools/list",
                serde_json::json!({
                    "_meta": {
                        "io.modelcontextprotocol/protocolVersion": "2026-07-28",
                        "io.modelcontextprotocol/clientInfo": {
                            "name": "gate-test", "version": "0.0.0"
                        },
                        "io.modelcontextprotocol/clientCapabilities": {},
                    }
                }),
            ))
            .await
            .expect("router responds");
        assert_eq!(
            modern.status(),
            StatusCode::OK,
            "a stateless 2026-07-28 tools/list must be served"
        );
    }
}
