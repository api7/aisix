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

// E2E: the scoped single-server MCP endpoint `/mcp/{server}` against a real
// gateway + etcd + two real MCP upstreams (official TypeScript SDK servers).
//
// Pinned contract:
//   - `initialize` reports the scoped server's registered name;
//   - `tools/list` returns the upstream's ORIGINAL tool names (no `server__`
//     prefix), filtered by the caller's ACL evaluated in namespaced form;
//   - `tools/call` accepts both the bare and the namespaced spelling; the
//     path — not the tool name — picks the server, so the same bare name
//     reaches different servers on different URLs;
//   - a foreign `other__` prefix stays pinned to the path's server;
//   - unknown and disabled servers are 404 (no fallback to the aggregate);
//   - the aggregated `/mcp` surface is unchanged (still namespaced).

const KEY_WILD = "sk-scoped-wild";
const KEY_ALPHA_ECHO = "sk-scoped-alpha-echo";
const KEY_BETA_ONLY = "sk-scoped-beta-only";

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

describe("mcp scoped endpoint e2e: /mcp/{server}", () => {
  let app: SpawnedApp | undefined;
  let alpha: McpUpstream | undefined;
  let beta: McpUpstream | undefined;
  let etcdReachable = false;
  let seed: SeedClient;

  const post = async (
    path: string,
    token: string,
    body: unknown,
  ): Promise<RpcReply> => {
    const res = await fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${token}`,
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

  /** Spec-faithful per-operation handshake (the endpoint is stateless). */
  const initialize = async (
    path: string,
    token: string,
  ): Promise<{ status: number; serverName?: string }> => {
    const init = await post(path, token, {
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: {
        protocolVersion: "2025-11-25",
        capabilities: {},
        clientInfo: { name: "mcp-scoped-e2e", version: "0.1" },
      },
    });
    if (init.status !== 200) return { status: init.status };
    await post(path, token, {
      jsonrpc: "2.0",
      method: "notifications/initialized",
    });
    return {
      status: init.status,
      serverName: init.json?.result?.serverInfo?.name,
    };
  };

  /** Sorted tool names visible to `token` at `path`, or an HTTP status. */
  const listToolNames = async (
    path: string,
    token: string,
  ): Promise<{ status: number; names?: string[] }> => {
    const { status } = await initialize(path, token);
    if (status !== 200) return { status };
    const r = await post(path, token, {
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
    token: string,
    name: string,
    text: string,
  ): Promise<{ ok: boolean; text?: string; error?: string }> => {
    await initialize(path, token);
    const r = await post(path, token, {
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

  /** True once `token` lists exactly `names` at `path` — propagation probe. */
  const listMatches = async (
    path: string,
    token: string,
    names: string[],
  ): Promise<boolean> => {
    const listed = await listToolNames(path, token);
    return (
      listed.status === 200 &&
      JSON.stringify(listed.names) === JSON.stringify(names)
    );
  };

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    alpha = await startMcpUpstream("alpha");
    beta = await startMcpUpstream("beta");
    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);

    await seed.update("mcp_servers", randomUUID(), {
      display_name: "alpha",
      url: alpha.url,
      enabled: true,
    });
    await seed.update("mcp_servers", randomUUID(), {
      display_name: "beta",
      url: beta.url,
      enabled: true,
    });
    await seed.update("mcp_servers", randomUUID(), {
      display_name: "dark",
      url: alpha.url,
      enabled: false,
    });

    const keyDoc = (
      plaintext: string,
      allowedTools: string[],
    ): Record<string, unknown> => ({
      key_hash: sha256(plaintext),
      allowed_models: [],
      mcp_access: { allow: allowedTools },
    });
    await seed.createApiKey(keyDoc(KEY_WILD, ["*"]));
    await seed.createApiKey(keyDoc(KEY_ALPHA_ECHO, ["alpha__echo"]));
    await seed.createApiKey(keyDoc(KEY_BETA_ONLY, ["beta__*"]));

    // Probe every key to its steady state so no later assertion races a row
    // that has not landed yet.
    await waitConfigPropagation(async () => {
      if (!(await listMatches("/mcp/alpha", KEY_WILD, ["echo", "reverse"]))) {
        return false;
      }
      if (!(await listMatches("/mcp/alpha", KEY_ALPHA_ECHO, ["echo"]))) {
        return false;
      }
      return listMatches("/mcp/beta", KEY_BETA_ONLY, ["echo", "reverse"]);
    });
  }, 60_000);

  afterAll(async () => {
    await app?.exit();
    await alpha?.close();
    await beta?.close();
  });

  test("initialize presents the scoped server, not the aggregate", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    const init = await initialize("/mcp/alpha", KEY_WILD);
    expect(init.status).toBe(200);
    expect(init.serverName).toBe("alpha");
  });

  test("tools/list returns original names; the aggregate stays namespaced", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    const scoped = await listToolNames("/mcp/alpha", KEY_WILD);
    expect(scoped.status).toBe(200);
    expect(scoped.names).toEqual(["echo", "reverse"]);

    // Regression: the aggregated endpoint is untouched by the scoped surface.
    const aggregated = await listToolNames("/mcp", KEY_WILD);
    expect(aggregated.status).toBe(200);
    expect(aggregated.names).toEqual([
      "alpha__echo",
      "alpha__reverse",
      "beta__echo",
      "beta__reverse",
    ]);
  });

  test("the path picks the server: the same bare name reaches different upstreams", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    const viaAlpha = await callTool("/mcp/alpha", KEY_WILD, "echo", "hi");
    expect(viaAlpha).toEqual({ ok: true, text: "alpha:hi" });

    const viaBeta = await callTool("/mcp/beta", KEY_WILD, "echo", "hi");
    expect(viaBeta).toEqual({ ok: true, text: "beta:hi" });

    // The namespaced spelling keeps working on the scoped URL.
    const namespaced = await callTool(
      "/mcp/alpha",
      KEY_WILD,
      "alpha__echo",
      "hi",
    );
    expect(namespaced).toEqual({ ok: true, text: "alpha:hi" });
  });

  test("a registered foreign prefix fails closed; an unregistered one is a bare name", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    // `beta` is a registered server, so `beta__echo` on /mcp/alpha is a
    // cross-server mistake: rejected, never routed to beta and never served
    // bare by alpha (the harness upstream would answer any name).
    const crossed = await callTool("/mcp/alpha", KEY_WILD, "beta__echo", "hi");
    expect(crossed.ok).toBe(false);
    expect(crossed.error).toContain("not available");

    // `ghost` is not registered, so `ghost__echo` is just a tool name that
    // contains the separator — it reaches alpha verbatim (the harness
    // upstream echoes for any non-`reverse` name, labelling the answer).
    const bare = await callTool("/mcp/alpha", KEY_WILD, "ghost__echo", "hi");
    expect(bare).toEqual({ ok: true, text: "alpha:hi" });
  });

  test("ACL keeps its namespaced meaning on the scoped endpoint", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    // A grant written as `alpha__echo` admits the bare `echo` here…
    const listed = await listToolNames("/mcp/alpha", KEY_ALPHA_ECHO);
    expect(listed.names).toEqual(["echo"]);
    const ok = await callTool("/mcp/alpha", KEY_ALPHA_ECHO, "echo", "hi");
    expect(ok).toEqual({ ok: true, text: "alpha:hi" });
    const denied = await callTool("/mcp/alpha", KEY_ALPHA_ECHO, "reverse", "hi");
    expect(denied.ok).toBe(false);
    expect(denied.error).toContain("not available");

    // …and a key scoped to beta reaches nothing on alpha's URL.
    const foreign = await listToolNames("/mcp/alpha", KEY_BETA_ONLY);
    expect(foreign.names).toEqual([]);
    const rejected = await callTool("/mcp/alpha", KEY_BETA_ONLY, "echo", "hi");
    expect(rejected.ok).toBe(false);
    expect(rejected.error).toContain("not available");
  });

  test("unknown and disabled servers are 404 with no fallback", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    const unknown = await initialize("/mcp/ghost", KEY_WILD);
    expect(unknown.status).toBe(404);

    const disabled = await initialize("/mcp/dark", KEY_WILD);
    expect(disabled.status).toBe(404);
  });
});
