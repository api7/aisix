import { createHash, randomUUID } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  spawnApp,
  startMcpUpstream,
  waitConfigPropagation,
  type McpUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: entry-level URL rewriting (`proxy.url_rewrites`) against a real
// gateway + etcd + a real MCP upstream.
//
// Pinned contract:
//   - the first rule whose `match` regex matches the path rewrites it; the
//     request then flows through the normal endpoint (auth, ACL, quota) as
//     if the client had sent the rewritten path;
//   - the flagship scenario: a client keeping its legacy per-server MCP URL
//     (`/mcp-servers/{service}/mcp`) and ORIGINAL tool names works end to
//     end through the rewritten `/mcp/{server}` endpoint;
//   - rewriting is generic, not MCP-specific (any path → any endpoint);
//   - a miss leaves the request untouched (canonical paths keep working,
//     unmatched legacy shapes 404);
//   - the query string survives the rewrite.

const KEY = "sk-url-rewrite-e2e";
const sha256 = (s: string) => createHash("sha256").update(s).digest("hex");

interface RpcReply {
  status: number;
  json?: {
    result?: {
      serverInfo?: { name?: string };
      tools?: Array<{ name: string }>;
      content?: Array<{ type: string; text?: string }>;
      isError?: boolean;
    };
    error?: { code: number; message: string };
  };
}

describe("url rewrite e2e: proxy.url_rewrites", () => {
  let app: SpawnedApp | undefined;
  let alpha: McpUpstream | undefined;
  let etcdReachable = false;

  const post = async (
    path: string,
    body: unknown,
  ): Promise<RpcReply> => {
    const res = await fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${KEY}`,
        "content-type": "application/json",
        accept: "application/json, text/event-stream",
      },
      body: JSON.stringify(body),
    });
    const text = await res.text();
    let json: RpcReply["json"];
    try {
      json = text ? JSON.parse(text) : undefined;
    } catch {
      json = undefined;
    }
    return { status: res.status, json };
  };

  const initialize = async (
    path: string,
  ): Promise<{ status: number; serverName?: string }> => {
    const init = await post(path, {
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: {
        protocolVersion: "2025-11-25",
        capabilities: {},
        clientInfo: { name: "url-rewrite-e2e", version: "0.1" },
      },
    });
    if (init.status !== 200) return { status: init.status };
    await post(path, { jsonrpc: "2.0", method: "notifications/initialized" });
    return {
      status: init.status,
      serverName: init.json?.result?.serverInfo?.name,
    };
  };

  const listToolNames = async (
    path: string,
  ): Promise<{ status: number; names?: string[] }> => {
    const { status } = await initialize(path);
    if (status !== 200) return { status };
    const r = await post(path, {
      jsonrpc: "2.0",
      id: 2,
      method: "tools/list",
      params: {},
    });
    const tools = r.json?.result?.tools;
    if (r.status !== 200 || !tools) return { status: r.status };
    return { status: r.status, names: tools.map((t) => t.name).sort() };
  };

  const callTool = async (
    path: string,
    name: string,
    text: string,
  ): Promise<{ ok: boolean; text?: string; error?: string }> => {
    await initialize(path);
    const r = await post(path, {
      jsonrpc: "2.0",
      id: 3,
      method: "tools/call",
      params: { name, arguments: { text } },
    });
    if (r.json?.error) return { ok: false, error: r.json.error.message };
    const result = r.json?.result;
    if (!result || result.isError) {
      return { ok: false, error: JSON.stringify(r.json ?? r.status) };
    }
    return { ok: true, text: result.content?.[0]?.text };
  };

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    alpha = await startMcpUpstream("alpha");
    app = await spawnApp({
      urlRewrites: [
        {
          name: "per-server-mcp-compat",
          match: "^/mcp-servers/([^/]+)/mcp$",
          rewrite: "/mcp/$1",
        },
        // Generic, non-MCP mapping: proves the layer rewrites any path
        // onto any endpoint.
        { match: "^/compat/health$", rewrite: "/livez" },
      ],
    });
    const seed = new SeedClient(new EtcdClient(), app.etcdPrefix);

    await seed.update("mcp_servers", randomUUID(), {
      display_name: "alpha",
      url: alpha.url,
      enabled: true,
    });
    await seed.createApiKey({
      key_hash: sha256(KEY),
      allowed_models: [],
      mcp_access: { allow: ["*"] },
    });

    await waitConfigPropagation(async () => {
      const listed = await listToolNames("/mcp-servers/alpha/mcp");
      return (
        listed.status === 200 &&
        JSON.stringify(listed.names) === JSON.stringify(["echo", "reverse"])
      );
    });
  }, 60_000);

  afterAll(async () => {
    await app?.exit();
    await alpha?.close();
  });

  test("legacy per-server MCP URL + original tool names work end to end", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    // The migration scenario in one flow: the client keeps its legacy URL
    // and its original tool name; the rewrite maps the URL onto
    // /mcp/{server}, which serves original names.
    const init = await initialize("/mcp-servers/alpha/mcp");
    expect(init.status).toBe(200);
    expect(init.serverName).toBe("alpha");

    const listed = await listToolNames("/mcp-servers/alpha/mcp");
    expect(listed.names).toEqual(["echo", "reverse"]);

    const called = await callTool("/mcp-servers/alpha/mcp", "echo", "hi");
    expect(called).toEqual({ ok: true, text: "alpha:hi" });
  });

  test("rewriting is generic: any path maps onto any endpoint", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    const res = await fetch(`${app.proxyUrl}/compat/health`);
    expect(res.status).toBe(200);

    // A request carrying a query string still matches and routes (`match`
    // never sees the query). Preservation of the query itself is pinned by
    // the `with_path_preserves_the_query_string` unit test — no proxy
    // endpoint echoes its query back for an e2e-level assertion.
    const withQuery = await fetch(`${app.proxyUrl}/compat/health?probe=1`);
    expect(withQuery.status).toBe(200);
  });

  test("a miss leaves the request untouched", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    // Canonical paths keep working alongside the legacy shapes…
    const canonical = await initialize("/mcp/alpha");
    expect(canonical.status).toBe(200);
    const livez = await fetch(`${app.proxyUrl}/livez`);
    expect(livez.status).toBe(200);

    // …and a legacy shape the rule does not match is NOT silently routed
    // anywhere (`backend_path` other than /mcp stays unserved).
    const unmatchedTail = await post("/mcp-servers/alpha/sse", {});
    expect(unmatchedTail.status).toBe(404);
    const unknownShape = await fetch(`${app.proxyUrl}/compat/other`);
    expect(unknownShape.status).toBe(404);
  });
});
