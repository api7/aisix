import { createHash, randomUUID } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  awaitWindowHeadroom,
  spawnApp,
  startMcpUpstream,
  waitConfigPropagation,
  type McpUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: per-(API key × MCP server) rate limiting against a real gateway +
// etcd + two real MCP upstreams (official TypeScript SDK servers).
//
// The key's `mcp_rate_limits` maps an MCP server name to that key's limits
// for it, so a runaway agent hammering one server cannot exhaust another's
// budget — nor another key's budget for the same server.
//
// Pinned contract:
//   - a key capped at rpm=2 on `alpha` gets 429 on its third alpha call
//     inside the window;
//   - `beta`, which the key sets no limit for, keeps serving;
//   - a second key with the identical alpha cap is unaffected by the first
//     key's burst (the counter is per key, not per server);
//   - a key with no `mcp_rate_limits` at all is never capped;
//   - `initialize` / `tools/list` keep answering while the cap is spent —
//     a client can always connect and enumerate.

const ALPHA_RPM = 2;

const KEY_CAPPED = "sk-mcp-rl-capped";
const KEY_CAPPED_TWIN = "sk-mcp-rl-capped-twin";
const KEY_UNCAPPED = "sk-mcp-rl-uncapped";

const sha256 = (s: string) => createHash("sha256").update(s).digest("hex");

interface RpcReply {
  status: number;
  json?: {
    result?: {
      tools?: Array<{ name: string }>;
      content?: Array<{ type: string; text?: string }>;
      isError?: boolean;
    };
    error?: { code: number; message: string };
  };
}

describe("mcp server rate limit e2e: per api key × mcp server", () => {
  let app: SpawnedApp | undefined;
  let alpha: McpUpstream | undefined;
  let beta: McpUpstream | undefined;
  let etcdReachable = false;
  let seed: SeedClient;

  const post = async (token: string, body: unknown): Promise<RpcReply> => {
    const res = await fetch(`${app!.proxyUrl}/mcp`, {
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
  const initialize = async (token: string): Promise<number> => {
    const init = await post(token, {
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: {
        protocolVersion: "2025-11-25",
        capabilities: {},
        clientInfo: { name: "mcp-rl-e2e", version: "0.1" },
      },
    });
    await post(token, { jsonrpc: "2.0", method: "notifications/initialized" });
    return init.status;
  };

  const listToolNames = async (
    token: string,
  ): Promise<{ status: number; names?: string[] }> => {
    const status = await initialize(token);
    if (status !== 200) return { status };
    const r = await post(token, {
      jsonrpc: "2.0",
      id: 2,
      method: "tools/list",
      params: {},
    });
    const tools = r.json?.result?.tools;
    if (r.status !== 200 || !tools) return { status: r.status };
    return { status: r.status, names: tools.map((t) => t.name).sort() };
  };

  /**
   * One `tools/call`, reported as the HTTP status plus the tool's text on
   * success. A rate-limited call never reaches the tool, so the caller
   * asserts on `status` alone.
   */
  const callTool = async (
    token: string,
    name: string,
  ): Promise<{ status: number; text?: string }> => {
    const r = await post(token, {
      jsonrpc: "2.0",
      id: 3,
      method: "tools/call",
      params: { name, arguments: { text: "hi" } },
    });
    return { status: r.status, text: r.json?.result?.content?.[0]?.text };
  };

  const ALL_TOOLS = [
    "alpha__echo",
    "alpha__reverse",
    "beta__echo",
    "beta__reverse",
  ];

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

    const keyDoc = (
      plaintext: string,
      extra: Record<string, unknown>,
    ): Record<string, unknown> => ({
      key_hash: sha256(plaintext),
      allowed_models: [],
      mcp_access: { allow: ["*"] },
      ...extra,
    });
    // Both capped keys carry the SAME alpha cap, so a shared counter would
    // show up as the twin inheriting the first key's exhaustion.
    await seed.createApiKey(
      keyDoc(KEY_CAPPED, { mcp_rate_limits: { alpha: { rpm: ALPHA_RPM } } }),
    );
    await seed.createApiKey(
      keyDoc(KEY_CAPPED_TWIN, {
        mcp_rate_limits: { alpha: { rpm: ALPHA_RPM } },
      }),
    );
    await seed.createApiKey(keyDoc(KEY_UNCAPPED, {}));

    // Probe every key to its steady state — `tools/list` is unmetered, so
    // the probe cannot spend any key's tool-call budget.
    await waitConfigPropagation(async () => {
      for (const token of [KEY_CAPPED, KEY_CAPPED_TWIN, KEY_UNCAPPED]) {
        const listed = await listToolNames(token);
        if (
          listed.status !== 200 ||
          JSON.stringify(listed.names) !== JSON.stringify(ALL_TOOLS)
        ) {
          return false;
        }
      }
      return true;
    });
  }, 60_000);

  afterAll(async () => {
    await app?.exit();
    await alpha?.close();
    await beta?.close();
  });

  test("the cap binds on its own server and leaves every other bucket alone", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    // The limiter buckets on fixed wall-clock minutes; keep the burst
    // inside one window so the 429 assertion can't straddle a roll-over.
    await awaitWindowHeadroom();

    for (let i = 0; i < ALPHA_RPM; i++) {
      const call = await callTool(KEY_CAPPED, "alpha__echo");
      expect(call.status).toBe(200);
      expect(call.text).toBe("alpha:hi");
    }

    const overLimit = await callTool(KEY_CAPPED, "alpha__echo");
    expect(overLimit.status).toBe(429);

    // A second tool on the SAME capped server shares that server's counter.
    const sameServerOtherTool = await callTool(KEY_CAPPED, "alpha__reverse");
    expect(sameServerOtherTool.status).toBe(429);

    // `beta` carries no entry for this key — its own counter is untouched.
    const otherServer = await callTool(KEY_CAPPED, "beta__echo");
    expect(otherServer.status).toBe(200);
    expect(otherServer.text).toBe("beta:hi");

    // The same cap on another key is a separate bucket.
    const twin = await callTool(KEY_CAPPED_TWIN, "alpha__echo");
    expect(twin.status).toBe(200);
    expect(twin.text).toBe("alpha:hi");

    // A key that sets no per-server limits is never capped.
    for (let i = 0; i < ALPHA_RPM + 2; i++) {
      const uncapped = await callTool(KEY_UNCAPPED, "alpha__echo");
      expect(uncapped.status).toBe(200);
    }

    // Handshake + discovery keep answering with the cap fully spent.
    const listed = await listToolNames(KEY_CAPPED);
    expect(listed.status).toBe(200);
    expect(listed.names).toEqual(ALL_TOOLS);
  }, 60_000);
});
