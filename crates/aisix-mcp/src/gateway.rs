//! The downstream-facing MCP gateway endpoint.
//!
//! [`McpGateway`] makes AISIX look like a single MCP server to a downstream
//! agent while fronting N registered upstream servers (each an [`McpBridge`]).
//! It is the other half of the dual role: an MCP *client* to each upstream
//! (via [`crate::RmcpBridge`]) and an MCP *server* to the agent (this type,
//! served over Streamable HTTP by [`streamable_http_service`]).
//!
//! Two operations, mirroring the upstream surface:
//! - `tools/list` fans out across every upstream and returns one aggregated
//!   list, each tool namespaced `server<SEP>tool`. An upstream that fails to
//!   list is skipped (its tools are simply absent), so one bad upstream does
//!   not blind the agent to the rest.
//! - `tools/call` strips the namespace prefix and routes to the owning
//!   upstream.
//!
//! A gateway may instead be **scoped** to a single upstream
//! ([`McpGateway::from_snapshot_scoped`], mounted at `/mcp/{server}`): it then
//! serves that server's tools under their original, un-namespaced names while
//! ACL decisions keep evaluating the namespaced form.
//!
//! The aggregator holds no per-request or per-session state, so governance
//! never depends on a transport session — which keeps it aligned with the
//! stateless direction of the MCP 2026-07-28 revision.
//!
//! Wiring this endpoint behind the gateway's auth / per-tool ACL / quota /
//! observability pipeline (and sourcing upstreams from the resource snapshot)
//! is the next step; this type takes an explicit set of upstreams and is not
//! yet mounted on any production listener.

use std::borrow::Cow;
use std::sync::Arc;

use rmcp::model::{
    CacheScope, CallToolRequestParams, CallToolResponse, CallToolResult, ContentBlock, ErrorData,
    ListToolsResult, PaginatedRequestParams, ProtocolVersion, ServerCapabilities, ServerInfo, Tool,
};
use rmcp::service::RequestContext;
use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
use rmcp::transport::streamable_http_server::{StreamableHttpServerConfig, StreamableHttpService};
use rmcp::{RoleServer, ServerHandler};

use aisix_core::models::{
    ApiKey, McpPolicy, McpPolicyScope, McpServerType,
};
use aisix_core::{AisixSnapshot, ResourceEntry};

use crate::bridge::{upstream_from_mcp_server, EphemeralBridge, McpBridge};
use crate::openapi::OpenApiBridge;

/// Separator between an upstream server's registered name and a tool name in
/// the aggregated namespace, e.g. `github__create_issue`. Server names must
/// not contain it; tool names may (we split on the first occurrence).
pub const TOOL_NAMESPACE_SEPARATOR: &str = "__";

/// Every protocol version `/mcp` serves, oldest first: the Streamable HTTP
/// generations (`2025-03-26` onward) plus the stateless `2026-07-28`
/// revision. The single source of truth for `initialize` negotiation,
/// `server/discover`, and the proxy layer's `MCP-Protocol-Version` header
/// gate — one list so the three can never drift.
pub const SUPPORTED_PROTOCOL_VERSIONS: &[ProtocolVersion] = &[
    ProtocolVersion::V_2025_03_26,
    ProtocolVersion::V_2025_06_18,
    ProtocolVersion::V_2025_11_25,
    ProtocolVersion::V_2026_07_28,
];

/// [`SUPPORTED_PROTOCOL_VERSIONS`] as plain strings, for callers that gate on
/// the `MCP-Protocol-Version` HTTP header without depending on rmcp types
/// (the proxy layer). Kept in lockstep by `supported_version_lists_agree`.
pub const SUPPORTED_PROTOCOL_VERSION_NAMES: &[&str] =
    &["2025-03-26", "2025-06-18", "2025-11-25", "2026-07-28"];

/// One registered upstream: its gateway-facing name and the live bridge to it.
struct NamedUpstream {
    name: String,
    bridge: Arc<dyn McpBridge>,
}

/// Strip `server`'s namespace prefix from `name`: `Some(bare)` when `name`
/// is `<server>__<bare>`, `None` otherwise. The one primitive both the
/// scoped gateway and the proxy's attribution peek use to interpret a tool
/// name on `/mcp/{server}`, so the two can never drift. Prefix matching is
/// by whole-string prefix (not first-separator split), so a server name
/// that itself ends in `_` still namespaces cleanly.
pub fn strip_server_prefix<'a>(server: &str, name: &'a str) -> Option<&'a str> {
    name.strip_prefix(server)
        .and_then(|rest| rest.strip_prefix(TOOL_NAMESPACE_SEPARATOR))
}

/// Which tools a gateway instance may expose and call, in the namespaced
/// `<server>__<tool>` form. Built per request from the caller's API key and
/// the environment's / the key's team's MCP access policies, so MCP tool
/// access is governed by the same key object as LLM access.
///
/// Three layers of identical shape contribute: the environment policy, the
/// key's team policy, and the key's own `mcp_access` block. A tool is
/// permitted only when **every present** allow layer admits it and **no**
/// deny pattern matches it — allow intersects, deny unions, and no layer can
/// widen what another one already narrowed. A layer that is absent or
/// disabled contributes nothing; with no allow layer at all the ACL grants
/// nothing, so MCP access is always granted explicitly.
#[derive(Clone)]
pub struct ToolAcl {
    /// Conjunctive allow layers: a tool must match every layer.
    allow: Vec<AllowLayer>,
    /// Deny patterns; any match rejects the tool, overriding every allow
    /// layer.
    deny: Vec<String>,
}

#[derive(Clone)]
enum AllowLayer {
    /// The layer admits every tool.
    All,
    /// The layer admits tools matching any of these single-`*` glob patterns.
    Patterns(Vec<String>),
}

impl AllowLayer {
    /// A bare `"*"` entry is folded into [`AllowLayer::All`]; every other
    /// list stays a pattern set (including the empty list, which admits
    /// nothing).
    fn from_patterns(patterns: &[String]) -> Self {
        if patterns.iter().any(|p| p == "*") {
            Self::All
        } else {
            Self::Patterns(patterns.to_vec())
        }
    }

    fn admits(&self, namespaced_tool: &str) -> bool {
        match self {
            Self::All => true,
            Self::Patterns(patterns) => patterns
                .iter()
                .any(|p| aisix_core::wildcard::wildcard_matches(p, namespaced_tool)),
        }
    }
}

impl ToolAcl {
    /// The unrestricted ACL — every aggregated tool is exposed. Gateways
    /// start here until scoped with [`McpGateway::with_tool_acl`].
    fn allow_all() -> Self {
        Self {
            allow: vec![AllowLayer::All],
            deny: Vec::new(),
        }
    }

    /// Narrow this ACL to the tools of `servers`, as an additional
    /// conjunctive allow layer: the result permits a tool only if the
    /// ACL already did AND the tool belongs to one of the named servers.
    ///
    /// Used for anonymous callers (AISIX-Cloud#1313), whose configured
    /// server allowlist is a ceiling on the bound principal rather than
    /// just an entry gate. Expressing it as a layer — instead of
    /// filtering `tools/list` at the endpoint — is what makes the
    /// aggregated and scoped endpoints agree: `permits` is the single
    /// point both listing and calling go through, and it evaluates the
    /// namespaced form on either endpoint.
    ///
    /// An empty list admits nothing, which is exactly right: an
    /// anonymous principal with no listed server has no tools.
    pub fn narrowed_to_servers(mut self, servers: &[String]) -> Self {
        // Built directly rather than via `from_patterns`: every entry
        // here is `<server>__*`, never a bare `*`, so the all-admitting
        // fold that helper performs must not apply.
        self.allow.push(AllowLayer::Patterns(
            servers
                .iter()
                .map(|s| format!("{s}{TOOL_NAMESPACE_SEPARATOR}*"))
                .collect(),
        ));
        self
    }

    /// Build a single-layer ACL from one allow list: an empty list grants no
    /// tools, a list containing `"*"` grants all, otherwise the listed
    /// patterns are matched as single-`*` globs (see [`ToolAcl::permits`]).
    /// No deny side — any caller serving external traffic resolves the full
    /// layered ACL with [`ToolAcl::resolve`] instead.
    pub fn from_allowed(allowed: Option<&[String]>) -> Self {
        Self {
            allow: vec![AllowLayer::from_patterns(allowed.unwrap_or(&[]))],
            deny: Vec::new(),
        }
    }

    /// Resolve the effective ACL for `key` from the `mcp_policies` in
    /// `snapshot`.
    ///
    /// Up to three layers apply, each an `allow`/`deny` pair: the
    /// environment policy, the key's team policy, and the key's own
    /// `mcp_access` block. Allow sides intersect (a layer can only narrow
    /// what the others left), deny sides union and always win. A layer that
    /// is absent — no row, a disabled row, or no `mcp_access` block —
    /// contributes neither side.
    ///
    /// With no allow layer at all the ACL grants nothing: MCP access is
    /// granted explicitly, never by the absence of configuration.
    pub fn resolve(snapshot: &AisixSnapshot, key: &ApiKey) -> Self {
        // Grant side: pick the governing row per scope deterministically
        // (lowest id wins) so a duplicated row — the writer enforces
        // uniqueness — can only ever produce a stable outcome. Deny side:
        // union across EVERY enabled row that applies to this key, so a
        // deny pattern can never disappear by losing a duplicate-row
        // tie-break.
        let mut env_policy: Option<Arc<ResourceEntry<McpPolicy>>> = None;
        let mut team_policy: Option<Arc<ResourceEntry<McpPolicy>>> = None;
        let mut deny: Vec<String> = Vec::new();
        for entry in snapshot.mcp_policies.entries() {
            if !entry.value.enabled {
                continue;
            }
            let slot = match entry.value.scope {
                McpPolicyScope::Env => &mut env_policy,
                McpPolicyScope::Team => {
                    let targets_key_team = key.team_id.is_some()
                        && key.team_id.as_deref() == entry.value.scope_ref.as_deref();
                    if !targets_key_team {
                        continue;
                    }
                    &mut team_policy
                }
            };
            deny.extend(entry.value.deny.iter().cloned());
            match slot {
                Some(current) if current.id <= entry.id => {}
                _ => *slot = Some(entry),
            }
        }

        let mut allow: Vec<AllowLayer> = Vec::new();
        for policy in [env_policy.as_ref(), team_policy.as_ref()].into_iter().flatten() {
            allow.push(AllowLayer::from_patterns(&policy.value.allow));
        }
        if let Some(access) = &key.mcp_access {
            allow.push(AllowLayer::from_patterns(&access.allow));
            deny.extend(access.deny.iter().cloned());
        }
        // Deny-by-default: an unconfigured key in an unconfigured
        // environment has no MCP access, rather than the empty conjunction's
        // "everything".
        if allow.is_empty() {
            allow.push(AllowLayer::Patterns(Vec::new()));
        }
        Self { allow, deny }
    }

    /// Whether `namespaced_tool` is permitted: every allow layer must admit
    /// it and no deny pattern may match it. Patterns are single-`*` globs:
    /// `"<server>__*"` covers every tool on that server, a pattern without a
    /// `*` matches exactly, and a bare `"*"` covers everything. Uses the same
    /// matcher as `ApiKey::can_access_tool`.
    pub fn permits(&self, namespaced_tool: &str) -> bool {
        self.allow.iter().all(|layer| layer.admits(namespaced_tool))
            && !self
                .deny
                .iter()
                .any(|p| aisix_core::wildcard::wildcard_matches(p, namespaced_tool))
    }
}

/// Aggregates N upstream MCP servers behind one downstream MCP server surface.
/// Cheap to clone (the upstream set is shared); the Streamable HTTP transport
/// clones it per session.
#[derive(Clone)]
pub struct McpGateway {
    upstreams: Arc<[NamedUpstream]>,
    tool_acl: ToolAcl,
    /// When set, this gateway serves exactly one upstream under its **original**
    /// tool names: `tools/list` strips the `<server>__` namespace prefix and
    /// `tools/call` accepts both the bare and the prefixed form. ACL decisions
    /// still evaluate the namespaced form, so per-tool grants keep one meaning
    /// across the aggregated and the scoped endpoint.
    scoped: Option<Arc<ScopedServer>>,
}

/// The single-server scope of a gateway built by
/// [`McpGateway::from_snapshot_scoped`].
struct ScopedServer {
    /// The scoped upstream's registered name — the namespace every ACL check
    /// re-applies and the name `initialize` reports.
    name: String,
    /// Every **other** registered server name, enabled or not. A `tools/call`
    /// whose name is prefixed with one of these is a cross-server mistake and
    /// fails closed rather than being silently served as a bare name.
    /// Disabled servers stay reserved so this scope's callable name surface
    /// does not shift when another server's `enabled` flag is toggled.
    foreign: std::collections::HashSet<String>,
}

impl McpGateway {
    /// Build a gateway over the given `(server_name, bridge)` upstreams.
    /// Registration order is the order tools are listed in.
    ///
    /// A name may only register once: a duplicate is dropped (the first
    /// registration wins) with a warning, rather than silently shadowing the
    /// later one and emitting duplicate tool names on the wire. Server names
    /// must not contain [`TOOL_NAMESPACE_SEPARATOR`].
    ///
    /// The gateway is **unrestricted** (every tool permitted) until scoped
    /// with [`McpGateway::with_tool_acl`]. Any caller that serves external
    /// traffic MUST scope it to the caller's key — the proxy `/mcp` mount is
    /// the single enforcement point and always does, via [`ToolAcl::resolve`].
    pub fn new(upstreams: impl IntoIterator<Item = (String, Arc<dyn McpBridge>)>) -> Self {
        let mut seen = std::collections::HashSet::new();
        let mut deduped = Vec::new();
        for (name, bridge) in upstreams {
            debug_assert!(
                !name.contains(TOOL_NAMESPACE_SEPARATOR),
                "upstream server name `{name}` must not contain the namespace \
                 separator `{TOOL_NAMESPACE_SEPARATOR}`"
            );
            if !seen.insert(name.clone()) {
                tracing::warn!(
                    upstream = %name,
                    "duplicate MCP upstream name; keeping the first registration, dropping this one"
                );
                continue;
            }
            deduped.push(NamedUpstream { name, bridge });
        }
        Self {
            upstreams: deduped.into(),
            tool_acl: ToolAcl::allow_all(),
            scoped: None,
        }
    }

    /// Scope this gateway to a per-tool [`ToolAcl`]: `tools/list` returns only
    /// permitted tools and `tools/call` rejects the rest. The mount builds this
    /// from the caller's API key.
    pub fn with_tool_acl(mut self, acl: ToolAcl) -> Self {
        self.tool_acl = acl;
        self
    }

    /// Build a gateway whose upstreams are the **enabled** `mcp_servers` in the
    /// snapshot: a `type: mcp` server is reached through an [`EphemeralBridge`]
    /// (connect per request), a `type: openapi` server through an
    /// [`OpenApiBridge`] that serves tools generated from its spec. Disabled
    /// servers are skipped. Registration order follows the snapshot's
    /// iteration order; duplicate names are deduped (first wins) by
    /// [`McpGateway::new`], though the Admin API already enforces uniqueness.
    pub fn from_snapshot(snapshot: &AisixSnapshot) -> Self {
        let upstreams = snapshot
            .mcp_servers
            .entries()
            .into_iter()
            .filter(|entry| entry.value.enabled)
            .map(|entry| {
                // Here, not inside one bridge constructor: `type: mcp` and
                // `type: openapi` rows share the credential fields, so the
                // cleartext warning covers both.
                crate::bridge::warn_cleartext_credential(&entry.value);
                let name = entry.value.name.clone();
                let bridge: Arc<dyn McpBridge> = match entry.value.server_type {
                    McpServerType::Mcp => {
                        let upstream = upstream_from_mcp_server(&entry.value);
                        Arc::new(EphemeralBridge::new(upstream))
                    }
                    McpServerType::Openapi => Arc::new(OpenApiBridge::new(entry)),
                };
                (name, bridge)
            });
        McpGateway::new(upstreams)
    }

    /// Build a gateway scoped to the single **enabled** `mcp_servers` entry
    /// named `server`, serving its tools under their original names (see
    /// [`McpGateway::scoped`]). Returns `None` when the server is not
    /// registered or is disabled — a disabled server is treated as absent,
    /// same as the aggregated endpoint skipping it.
    pub fn from_snapshot_scoped(snapshot: &AisixSnapshot, server: &str) -> Option<Self> {
        let entry = snapshot.mcp_servers.get_by_name(server)?;
        if !entry.value.enabled {
            return None;
        }
        crate::bridge::warn_cleartext_credential(&entry.value);
        let name = entry.value.name.clone();
        let bridge: Arc<dyn McpBridge> = match entry.value.server_type {
            McpServerType::Mcp => {
                let upstream = upstream_from_mcp_server(&entry.value);
                Arc::new(EphemeralBridge::new(upstream))
            }
            McpServerType::Openapi => Arc::new(OpenApiBridge::new(entry)),
        };
        let foreign = snapshot
            .mcp_servers
            .entries()
            .into_iter()
            .filter(|e| e.value.name != name)
            .map(|e| e.value.name.clone())
            .collect();
        let mut gateway = McpGateway::new([(name.clone(), bridge)]);
        gateway.scoped = Some(Arc::new(ScopedServer { name, foreign }));
        Some(gateway)
    }

    fn find(&self, server: &str) -> Option<&Arc<dyn McpBridge>> {
        self.upstreams
            .iter()
            .find(|u| u.name == server)
            .map(|u| &u.bridge)
    }
}

impl ServerHandler for McpGateway {
    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, ErrorData> {
        // Fan out concurrently; each upstream call is already deadline-bounded
        // by its bridge, so a slow upstream cannot stall the aggregate.
        let listed = futures::future::join_all(
            self.upstreams
                .iter()
                .map(|u| async move { (u.name.as_str(), u.bridge.list_tools().await) }),
        )
        .await;

        let mut tools = Vec::new();
        for (server, result) in listed {
            match result {
                Ok(upstream_tools) => {
                    tools.extend(upstream_tools.into_iter().map(|t| prefixed_tool(server, t)));
                }
                Err(error) => {
                    // Degrade gracefully: drop this upstream's tools, keep the
                    // rest. Detail is logged server-side (never client-visible).
                    tracing::warn!(
                        upstream = server,
                        error = %error,
                        "skipping upstream in tools/list: list_tools failed"
                    );
                }
            }
        }
        // Per-tool ACL: expose only the tools this caller's key permits.
        tools.retain(|tool| self.tool_acl.permits(tool.name.as_ref()));
        // A scoped gateway serves its single upstream's tools under their
        // original names — the namespace prefix exists to disambiguate the
        // aggregate, and a single-server endpoint has nothing to disambiguate.
        // ACL filtering above ran on the namespaced form. Strip only when the
        // bare name round-trips through `call_tool`'s parsing: a literal
        // upstream name that itself starts with a registered server's prefix
        // would be re-stripped (or fail closed) if advertised bare, so those
        // stay namespaced — that spelling is the one `call_tool` accepts.
        if let Some(scoped) = &self.scoped {
            for tool in &mut tools {
                if let Some(bare) = strip_server_prefix(&scoped.name, tool.name.as_ref()) {
                    let round_trips = strip_server_prefix(&scoped.name, bare).is_none()
                        && !scoped
                            .foreign
                            .iter()
                            .any(|f| strip_server_prefix(f, bare).is_some());
                    if round_trips {
                        tool.name = Cow::Owned(bare.to_string());
                    }
                }
            }
        }
        // Cache hints (SEP-2549, required on 2026-07-28 list results) are a
        // wire-level statement only — the gateway runs no cache engine. The
        // honest values here are "do not cache": the list is filtered by the
        // CALLER's per-key ACL (so it is `private` to that authorization
        // context), and upstream tool sets can change between requests (so
        // its freshness window is zero).
        //
        // Version-gated because rmcp only strips `resultType` for legacy
        // peers, not these fields — setting them unconditionally would add
        // two fields legacy clients have never seen. Modern requests carry
        // the version in `_meta`; legacy sessions fall back to the
        // handshake's negotiated version. (ISO `YYYY-MM-DD` versions compare
        // lexically the same as chronologically.)
        let result = ListToolsResult::with_all_items(tools);
        let modern = context
            .protocol_version()
            .is_some_and(|v| v.as_str() >= ProtocolVersion::V_2026_07_28.as_str());
        Ok(if modern {
            result.with_ttl_ms(0).with_cache_scope(CacheScope::Private)
        } else {
            result
        })
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, ErrorData> {
        // Resolve `(server, tool, namespaced)` from the caller's tool name.
        // Scoped: the name is the upstream's original one, but a caller that
        // sends the namespaced form anyway (an aggregated-endpoint client
        // pointed at the scoped URL) keeps working — `<scope>__x` reduces to
        // `x`. (An upstream tool literally named `<scope>__x` is therefore
        // only callable as `<scope>__<scope>__x` — one strip, own prefix
        // first; `tools/list` advertises exactly that spelling.) A name
        // prefixed with a *different* registered server's name fails closed:
        // the scoped endpoint never cross-routes, and silently serving it as
        // a bare name would mask the caller's mistake. An unregistered
        // prefix stays a bare name, since tool names may legitimately
        // contain the separator.
        let request_name = request.name.as_ref();
        let (server, tool, namespaced): (&str, &str, Cow<'_, str>) = match &self.scoped {
            Some(scoped) => {
                let scope = scoped.name.as_str();
                let bare = match strip_server_prefix(scope, request_name) {
                    Some(rest) => rest,
                    None if scoped
                        .foreign
                        .iter()
                        .any(|f| strip_server_prefix(f, request_name).is_some()) =>
                    {
                        // Same neutral wording as the ACL reject: don't
                        // confirm what the other server serves.
                        return Err(ErrorData::invalid_params(
                            format!("tool '{request_name}' is not available"),
                            None,
                        ));
                    }
                    None => request_name,
                };
                (
                    scope,
                    bare,
                    Cow::Owned(format!("{scope}{TOOL_NAMESPACE_SEPARATOR}{bare}")),
                )
            }
            None => {
                let (server, tool) = request_name
                    .split_once(TOOL_NAMESPACE_SEPARATOR)
                    .ok_or_else(|| {
                        ErrorData::invalid_params(
                            format!(
                                "tool name '{request_name}' is missing a 'server__tool' prefix"
                            ),
                            None,
                        )
                    })?;
                (server, tool, Cow::Borrowed(request_name))
            }
        };

        // Per-tool ACL: reject a call the caller's key doesn't permit. A
        // disallowed tool is also absent from `tools/list`, so this is
        // defense-in-depth; the message stays neutral and does not reveal
        // whether the tool exists upstream. The check runs on the namespaced
        // form so grants mean the same thing on every endpoint; the message
        // echoes the caller's own spelling.
        if !self.tool_acl.permits(&namespaced) {
            return Err(ErrorData::invalid_params(
                format!("tool '{request_name}' is not available"),
                None,
            ));
        }

        let bridge = self.find(server).ok_or_else(|| {
            ErrorData::invalid_params(format!("unknown MCP server '{server}'"), None)
        })?;

        let arguments = request
            .arguments
            .map(serde_json::Value::Object)
            .unwrap_or(serde_json::Value::Null);

        let result = bridge.call_tool(tool, arguments).await.map_err(|error| {
            // Generic client-facing message; the upstream's detail (which may
            // include its URL) is logged server-side, not surfaced to the agent.
            tracing::warn!(
                upstream = server,
                tool = tool,
                error = %error,
                "upstream tools/call failed"
            );
            ErrorData::internal_error(
                format!("upstream MCP server '{server}' failed to call tool"),
                None,
            )
        })?;

        // Always a final (`Complete`) response: the gateway never asks the
        // downstream agent for more input (MRTR stays un-relayed — see
        // `RmcpBridge::call_tool`), so `InputRequired` is never produced here.
        into_call_tool_result(result).map(CallToolResponse::from)
    }

    /// The protocol versions this endpoint implements, replacing the SDK
    /// default (`ProtocolVersion::KNOWN_VERSIONS`, which would advertise
    /// every version rmcp has ever heard of — including ones this endpoint
    /// does not serve). Consulted by rmcp for `initialize` negotiation (an
    /// exact match is echoed; anything else falls back to
    /// `get_info().protocol_version`) and advertised verbatim in
    /// `server/discover`.
    ///
    /// `2024-11-05` is deliberately absent: that generation's transport is
    /// HTTP+SSE, which this Streamable-HTTP-only endpoint has never served.
    fn supported_protocol_versions(&self) -> Cow<'static, [ProtocolVersion]> {
        Cow::Borrowed(SUPPORTED_PROTOCOL_VERSIONS)
    }

    fn get_info(&self) -> ServerInfo {
        let mut info = ServerInfo::default();
        // Pinned, not `ProtocolVersion::default()`: this is the version a
        // legacy `initialize` falls back to when the client requests one we
        // don't list in `supported_protocol_versions` — the same answer the
        // endpoint has always given. Riding the SDK's `LATEST` alias would
        // silently move this fallback whenever rmcp bumps it.
        info.protocol_version = ProtocolVersion::V_2025_11_25;
        info.capabilities = ServerCapabilities::builder().enable_tools().build();
        match &self.scoped {
            // The scoped endpoint presents as the upstream server itself, so
            // `initialize` reports that server's registered name.
            Some(scoped) => {
                info.server_info.name = scoped.name.clone();
                info.instructions = Some(format!(
                    "AISIX MCP gateway: serves the tools of MCP server `{}` \
                     under their original names.",
                    scoped.name
                ));
            }
            None => {
                info.instructions = Some(
                    "AISIX MCP gateway: aggregates tools from registered upstream MCP \
                     servers, namespaced as `server__tool`."
                        .to_string(),
                );
            }
        }
        info
    }
}

/// Build the Streamable HTTP service for this gateway, ready to nest in axum
/// at `/mcp`.
///
/// Configured stateless (no sticky session, JSON responses): the aggregator
/// keeps no per-session state, so the endpoint can sit behind a plain load
/// balancer — matching the MCP 2026-07-28 transport direction.
///
/// `request_body_limit_bytes` is the deployment's request-body cap
/// (`0` = unlimited, the same convention as everywhere else in the data
/// plane). It must be threaded in because rmcp 3.x added its OWN inbound
/// cap (4 MiB default) inside this service — beneath the gateway's limit
/// middleware — which would silently override any configured limit above
/// 4 MiB (and the documented unlimited mode) with a plain-text 413.
pub fn streamable_http_service(
    gateway: McpGateway,
    request_body_limit_bytes: usize,
) -> StreamableHttpService<McpGateway, LocalSessionManager> {
    let mut config = StreamableHttpServerConfig::default();
    config.max_request_body_bytes = if request_body_limit_bytes == 0 {
        usize::MAX
    } else {
        request_body_limit_bytes
    };
    // rmcp 3.x rename of `stateful_mode` (same semantics for legacy
    // protocol versions; 2026-07-28 requests are stateless regardless, per
    // SEP-2567). Kept `false`: the aggregator holds no session state, so
    // legacy clients get sessionless serving too and the endpoint stays
    // safe behind a plain load balancer.
    config.legacy_session_mode = false;
    config.json_response = true;
    // Disable rmcp's `Host`-header allowlist. Its default
    // (`localhost`/`127.0.0.1`/`::1`) is a DNS-rebinding guard for
    // browser-driven local servers — it 403s every request whose `Host` is
    // the deployment's real DNS name. This endpoint is not browser-driven: it
    // is reached server-to-server by agents and is gated by the AISIX API key,
    // which is the real access control. An empty allowlist accepts any `Host`;
    // the request is still authenticated upstream of this service. (Operators
    // who want Host pinning can layer it at their ingress.)
    config.allowed_hosts = Vec::new();
    StreamableHttpService::new(
        move || Ok(gateway.clone()),
        Arc::new(LocalSessionManager::default()),
        config,
    )
}

/// Namespace an upstream tool: `server<SEP>tool`, preserving its schema and
/// (optional) description.
fn prefixed_tool(server: &str, tool: crate::McpTool) -> Tool {
    let schema = match tool.input_schema {
        serde_json::Value::Object(map) => map,
        // A non-object schema is malformed per JSON Schema; advertise an empty
        // object rather than dropping the tool.
        _ => serde_json::Map::new(),
    };
    Tool::new_with_raw(
        format!("{server}{TOOL_NAMESPACE_SEPARATOR}{}", tool.name),
        tool.description.map(Cow::Owned),
        schema,
    )
}

/// Map our [`crate::McpToolResult`] back onto rmcp's `CallToolResult`,
/// preserving the upstream's tool-level error flag (a tool-level error is
/// propagated as `Ok(error_result)`, not turned into a protocol error).
fn into_call_tool_result(result: crate::McpToolResult) -> Result<CallToolResult, ErrorData> {
    let content: Vec<ContentBlock> = serde_json::from_value(result.content).map_err(|e| {
        ErrorData::internal_error(format!("malformed tool content from upstream: {e}"), None)
    })?;
    let mut call_result = if result.is_error {
        CallToolResult::error(content)
    } else {
        CallToolResult::success(content)
    };
    call_result.structured_content = result.structured_content;
    Ok(call_result)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rmcp-typed list (initialize negotiation, `server/discover`) and
    /// the string list (the proxy's `MCP-Protocol-Version` header gate) are
    /// two spellings of ONE decision; a version added to either alone is a
    /// drift bug.
    #[test]
    fn supported_version_lists_agree() {
        let typed: Vec<&str> = SUPPORTED_PROTOCOL_VERSIONS
            .iter()
            .map(|v| v.as_str())
            .collect();
        assert_eq!(typed, SUPPORTED_PROTOCOL_VERSION_NAMES);
    }

    /// `2024-11-05` (the HTTP+SSE generation) must stay excluded: this
    /// endpoint serves Streamable HTTP only.
    #[test]
    fn http_sse_generation_stays_unsupported() {
        assert!(!SUPPORTED_PROTOCOL_VERSION_NAMES.contains(&"2024-11-05"));
        assert!(!SUPPORTED_PROTOCOL_VERSIONS.contains(&ProtocolVersion::V_2024_11_05));
    }

    /// The anonymous ceiling intersects — it can only ever remove tools.
    /// A principal whose own grant is `*` must still be confined to the
    /// listed servers, which is what stops an anonymous caller from
    /// reaching an unlisted server by naming `<server>__<tool>` on the
    /// aggregated endpoint while its scoped entry is closed.
    #[test]
    fn server_ceiling_narrows_a_wide_grant() {
        let wide = ToolAcl::from_allowed(Some(&["*".to_string()]));
        assert!(wide.permits("kb__search"));

        let capped = ToolAcl::from_allowed(Some(&["*".to_string()]))
            .narrowed_to_servers(&["docs".to_string()]);
        assert!(capped.permits("docs__search"));
        assert!(!capped.permits("kb__search"));
        // A bare tool name belongs to no server and is never admitted.
        assert!(!capped.permits("search"));
        // The `__` boundary is respected: a server whose name merely
        // starts with an allowed one does not inherit its grant.
        assert!(!capped.permits("docsecret__search"));
    }

    /// The ceiling never widens: a narrow grant stays narrow even when
    /// the allowlist names more servers than the key can reach.
    #[test]
    fn server_ceiling_cannot_widen_a_grant() {
        let acl = ToolAcl::from_allowed(Some(&["docs__search".to_string()]))
            .narrowed_to_servers(&["docs".to_string(), "kb".to_string()]);
        assert!(acl.permits("docs__search"));
        assert!(!acl.permits("docs__write"));
        assert!(!acl.permits("kb__search"));
    }

    /// An empty allowlist admits nothing — an anonymous principal with
    /// no listed server has no tools, rather than all of them.
    #[test]
    fn server_ceiling_with_no_servers_admits_nothing() {
        let acl = ToolAcl::from_allowed(Some(&["*".to_string()])).narrowed_to_servers(&[]);
        assert!(!acl.permits("docs__search"));
        assert!(!acl.permits("anything"));
    }

    /// The exact served set, as literals: growing or shrinking it is a
    /// deliberate protocol-surface decision that must show up as a failing
    /// test, not ride along inside a refactor that edits the constants.
    /// (The lockstep test above only proves the two constants AGREE — it
    /// would pass if a fifth version were added to both.)
    #[test]
    fn supported_version_set_is_pinned() {
        assert_eq!(
            SUPPORTED_PROTOCOL_VERSION_NAMES,
            &["2025-03-26", "2025-06-18", "2025-11-25", "2026-07-28"],
            "update this pin together with the negotiation/discover/header-gate \
             docs and the tracking issue when the served set changes"
        );
    }
}
