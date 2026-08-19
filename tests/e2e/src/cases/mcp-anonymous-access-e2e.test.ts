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

// E2E: anonymous access to `/mcp` entries (AISIX-Cloud#1313) against a real
// gateway + etcd + two real MCP upstreams.
//
// The environment offers `docs` anonymously and the aggregated endpoint too;
// `kb` is registered but NOT offered. The bound principal deliberately holds
// a wildcard tool grant, which is what makes the interesting assertions
// meaningful: everything that keeps an anonymous caller away from `kb` comes
// from the anonymous configuration, not from the key.
//
// Pinned contract:
//   - a request with NO credential is served on a listed entry, as the bound
//     principal (its ACL, limits and attribution apply);
//   - the server list is a CEILING, not just an entry gate: the aggregated
//     endpoint hides `kb__*` and refuses to call it, even though the
//     principal's own grant is `*`;
//   - an unlisted-but-registered server is indistinguishable from an unknown
//     one (401, never 404);
//   - a credential that is offered and fails NEVER downgrades to anonymous;
//   - a valid credential keeps its own identity and wider reach on the same
//     entries.

const KEY_WILD = "sk-anon-e2e-wild";
const ANON_PRINCIPAL_SECRET = `sk-anon-e2e-principal-${randomUUID()}`;

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

describe("mcp anonymous access e2e", () => {
  let app: SpawnedApp | undefined;
  let docs: McpUpstream | undefined;
  let kb: McpUpstream | undefined;
  let etcdReachable = false;
  let seed: SeedClient;
  const settingsId = randomUUID();

  /** POST a JSON-RPC body; `token` omitted means no credential at all. */
  const post = async (
    path: string,
    token: string | undefined,
    body: unknown,
  ): Promise<RpcReply> => {
    const headers: Record<string, string> = {
      "content-type": "application/json",
      accept: "application/json, text/event-stream",
    };
    if (token !== undefined) headers.authorization = `Bearer ${token}`;
    const res = await fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers,
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
    token: string | undefined,
  ): Promise<number> => {
    const init = await post(path, token, {
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: {
        protocolVersion: "2025-11-25",
        capabilities: {},
        clientInfo: { name: "mcp-anon-e2e", version: "0.1" },
      },
    });
    if (init.status === 200) {
      await post(path, token, {
        jsonrpc: "2.0",
        method: "notifications/initialized",
      });
    }
    return init.status;
  };

  const listToolNames = async (
    path: string,
    token: string | undefined,
  ): Promise<{ status: number; names?: string[] }> => {
    const status = await initialize(path, token);
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
    token: string | undefined,
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

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    docs = await startMcpUpstream("docs");
    kb = await startMcpUpstream("kb");
    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);

    await seed.update("mcp_servers", randomUUID(), {
      display_name: "docs",
      url: docs.url,
      enabled: true,
    });
    await seed.update("mcp_servers", randomUUID(), {
      display_name: "kb",
      url: kb.url,
      enabled: true,
    });

    // The anonymous principal holds a WILDCARD grant on purpose: if the
    // server allowlist were only an entry gate, this key would reach `kb`
    // through the aggregated endpoint.
    const principal = await seed.createApiKey({
      key_hash: sha256(ANON_PRINCIPAL_SECRET),
      allowed_models: [],
      mcp_access: { allow: ["*"] },
    });
    await seed.createApiKey({
      key_hash: sha256(KEY_WILD),
      allowed_models: [],
      mcp_access: { allow: ["*"] },
    });

    await seed.update("mcp_auth_settings", settingsId, {
      anonymous: {
        api_key_id: principal.id,
        // The e2e client connects over loopback; both families, since
        // `localhost` may resolve either way.
        source_cidrs: ["127.0.0.0/8", "::1/128"],
        servers: ["docs"],
        aggregate_entry: true,
      },
    });

    await waitConfigPropagation(async () => {
      const anon = await listToolNames("/mcp/docs", undefined);
      if (anon.status !== 200) return false;
      if (JSON.stringify(anon.names) !== JSON.stringify(["echo", "reverse"])) {
        return false;
      }
      const authed = await listToolNames("/mcp", KEY_WILD);
      return (
        authed.status === 200 &&
        JSON.stringify(authed.names) ===
          JSON.stringify([
            "docs__echo",
            "docs__reverse",
            "kb__echo",
            "kb__reverse",
          ])
      );
    });
  }, 60_000);

  afterAll(async () => {
    await app?.exit();
    await docs?.close();
    await kb?.close();
  });

  test("a listed entry serves a caller with no credential", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    const listed = await listToolNames("/mcp/docs", undefined);
    expect(listed.status).toBe(200);
    // The scoped endpoint keeps the upstream's original tool names.
    expect(listed.names).toEqual(["echo", "reverse"]);

    const called = await callTool("/mcp/docs", undefined, "echo", "hello");
    expect(called.ok).toBe(true);
    expect(called.text).toContain("hello");
  });

  test("the server list caps the principal on the aggregated entry", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    // The principal's own grant is `*` and `kb` is registered and enabled,
    // yet an anonymous caller must see neither its tools nor be able to
    // name one — otherwise closing `/mcp/kb` would be cosmetic.
    const listed = await listToolNames("/mcp", undefined);
    expect(listed.status).toBe(200);
    expect(listed.names).toEqual(["docs__echo", "docs__reverse"]);

    const called = await callTool("/mcp", undefined, "kb__echo", "hello");
    expect(called.ok).toBe(false);
  });

  test("an unlisted server is indistinguishable from an unknown one", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    // `kb` exists; `ghost` does not. Both must answer 401 — a 404 for one
    // of them would let an anonymous prober map the registered set.
    for (const server of ["kb", "ghost"]) {
      expect(await initialize(`/mcp/${server}`, undefined)).toBe(401);
    }
  });

  test("a failed credential never downgrades to anonymous", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    // Each of these would succeed if it were simply dropped and the
    // request re-read as credential-less.
    for (const bad of ["sk-not-a-real-key", ANON_PRINCIPAL_SECRET.slice(0, -1)]) {
      expect(await initialize("/mcp/docs", bad)).toBe(401);
      expect(await initialize("/mcp", bad)).toBe(401);
    }
  });

  test("a valid credential keeps its own identity and reach", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    // Anonymous configuration does not shadow the authenticated path: the
    // same entries stay fully reachable for a real key.
    const listed = await listToolNames("/mcp", KEY_WILD);
    expect(listed.status).toBe(200);
    expect(listed.names).toEqual([
      "docs__echo",
      "docs__reverse",
      "kb__echo",
      "kb__reverse",
    ]);

    const called = await callTool("/mcp/kb", KEY_WILD, "echo", "hello");
    expect(called.ok).toBe(true);
  });

  test("clearing the settings row closes anonymous access", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    // cp-api projects a delete when an operator turns the feature off;
    // the watch path must pick it up without a resync.
    await seed.delete("mcp_auth_settings", settingsId);

    await waitConfigPropagation(
      async () => (await initialize("/mcp/docs", undefined)) === 401,
    );
    expect(await initialize("/mcp", undefined)).toBe(401);
    // The authenticated path is unaffected by the removal.
    expect(await initialize("/mcp", KEY_WILD)).toBe(200);
  });
});
