import { createHash, randomUUID } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  spawnApp,
  startMcpUpstream,
  waitConfigPropagation,
  type McpUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: what the DP logs for a `/mcp` request, against a real gateway + etcd
// + a real MCP upstream (official TypeScript SDK server). MCP tunnels every
// operation through one `POST /mcp`, so before #1181 the access line said
// only `method="POST" path="/mcp" status=200` — identical for a handshake, a
// tool call, and a `tools/list` the ACL had emptied.
//
// Pinned contract:
//   - the access line names the JSON-RPC method, the called tool, and (for
//     `tools/list`) the tool counts on both sides of ACL filtering;
//   - a `tools/list` emptied by the ACL warns once, naming WHICH of the two
//     misconfigurations it was: no grant at all, or a grant matching nothing;
//   - an upstream whose tools all survive warns not at all;
//   - none of this changes the response: an emptied list is still 200 with
//     `tools: []`.

const KEY_NO_GRANT = "sk-mcp-log-no-grant";
const KEY_NO_MATCH = "sk-mcp-log-no-match";
const KEY_GRANTED = "sk-mcp-log-granted";

const NO_GRANT_WARN =
  "mcp tools/list returned no tools: no MCP access policy or key-level grant applies to this caller";
const NO_MATCH_WARN =
  "mcp tools/list returned no tools: no granted pattern matched any upstream tool";

const sha256 = (s: string) => createHash("sha256").update(s).digest("hex");

interface RpcReply {
  status: number;
  json?: {
    result?: { tools?: Array<{ name: string }> };
    error?: { code: number; message: string };
  };
}

describe("mcp request logging e2e: JSON-RPC method, tool counts, ACL warning", () => {
  let app: SpawnedApp | undefined;
  let alpha: McpUpstream | undefined;
  let etcdReachable = false;
  const keyIds: Record<string, string> = {};

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
  const initialize = (token: string): Promise<RpcReply> =>
    post(token, {
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: {
        protocolVersion: "2025-11-25",
        capabilities: {},
        clientInfo: { name: "mcp-request-logging-e2e", version: "0.1" },
      },
    });

  const listTools = async (token: string): Promise<RpcReply> => {
    await initialize(token);
    return post(token, {
      jsonrpc: "2.0",
      id: 2,
      method: "tools/list",
      params: {},
    });
  };

  /** Lines of DP output matching `pred`, after letting the pipe settle. */
  const matchingLines = async (
    pred: (line: string) => boolean,
  ): Promise<string[]> => {
    await new Promise((resolve) => setTimeout(resolve, 250));
    return app!.output().split("\n").filter(pred);
  };

  const waitForLine = async (
    pred: (line: string) => boolean,
    what: string,
  ): Promise<string> => {
    const deadline = Date.now() + 5_000;
    let last = "";
    while (Date.now() < deadline) {
      last = app!.output();
      const hit = last.split("\n").find(pred);
      if (hit) return hit;
      await new Promise((resolve) => setTimeout(resolve, 50));
    }
    throw new Error(`timed out waiting for ${what}; DP output was:\n${last}`);
  };

  /** The access line this key's `method` request wrote. */
  const accessLine = (token: string, method: string): Promise<string> =>
    waitForLine(
      (l) =>
        l.includes("proxy request completed") &&
        l.includes(`api_key_id="${keyIds[token]}"`) &&
        l.includes(`mcp_method="${method}"`),
      `the ${method} access-log line for ${token}`,
    );

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    alpha = await startMcpUpstream("alpha");
    // The access log is an `info` line; the suite default is `warn`.
    app = await spawnApp({ logLevel: "info" });
    const seed = new SeedClient(etcd, app.etcdPrefix);

    await seed.update("mcp_servers", randomUUID(), {
      display_name: "alpha",
      url: alpha.url,
      enabled: true,
    });

    // No `mcp_policies` at all, so the key's own block is the only possible
    // grant — which is what makes the "no grant applies" case reachable.
    const seedKey = async (plaintext: string, extra: Record<string, unknown>) => {
      const { id } = await seed.createApiKey({
        key_hash: sha256(plaintext),
        allowed_models: [],
        ...extra,
      });
      keyIds[plaintext] = id;
    };
    await seedKey(KEY_NO_GRANT, {});
    await seedKey(KEY_NO_MATCH, { mcp_access: { allow: ["ghost__*"] } });
    await seedKey(KEY_GRANTED, { mcp_access: { allow: ["*"] } });

    // Gate on the LAST key seeded authenticating, not on a `tools/list`:
    // the keys are written after the server row, so this one condition
    // implies the whole seed set has landed, and it is not the behaviour
    // under test (which would time out instead of failing an assertion).
    const probe = new ProxyClient(app.proxyUrl, KEY_GRANTED);
    await waitConfigPropagation(async () => (await probe.listModels()).status === 200);
  }, 60_000);

  afterAll(async () => {
    await app?.exit();
    await alpha?.close();
  });

  test("a key with no grant: counts on the access line, one warn naming the reason", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    const listed = await listTools(KEY_NO_GRANT);
    // The response behaviour is deliberate and unchanged: fail-closed, but
    // as an empty list rather than an error.
    expect(listed.status).toBe(200);
    expect(listed.json?.result?.tools).toEqual([]);

    const line = await accessLine(KEY_NO_GRANT, "tools/list");
    expect(line).toContain("tools_total=2");
    expect(line).toContain("tools_returned=0");

    const warns = await matchingLines(
      (l) => l.includes(NO_GRANT_WARN) && l.includes(`api_key_id="${keyIds[KEY_NO_GRANT]}"`),
    );
    expect(warns).toHaveLength(1);
    expect(warns[0]).toContain("upstream_tools=2");
  });

  test("a grant that matches nothing is a different warning", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    const listed = await listTools(KEY_NO_MATCH);
    expect(listed.status).toBe(200);
    expect(listed.json?.result?.tools).toEqual([]);

    const line = await accessLine(KEY_NO_MATCH, "tools/list");
    expect(line).toContain("tools_total=2");
    expect(line).toContain("tools_returned=0");

    const warns = await matchingLines(
      (l) => l.includes(NO_MATCH_WARN) && l.includes(`api_key_id="${keyIds[KEY_NO_MATCH]}"`),
    );
    expect(warns).toHaveLength(1);
    // The two wordings are exclusive: this key HAS a grant.
    const wrong = await matchingLines(
      (l) => l.includes(NO_GRANT_WARN) && l.includes(`api_key_id="${keyIds[KEY_NO_MATCH]}"`),
    );
    expect(wrong).toEqual([]);
  });

  test("a granted key logs both counts and warns not at all", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    const listed = await listTools(KEY_GRANTED);
    expect(listed.json?.result?.tools).toHaveLength(2);

    const line = await accessLine(KEY_GRANTED, "tools/list");
    expect(line).toContain("tools_total=2");
    expect(line).toContain("tools_returned=2");

    const warns = await matchingLines(
      (l) =>
        l.includes("returned no tools") &&
        l.includes(`api_key_id="${keyIds[KEY_GRANTED]}"`),
    );
    expect(warns).toEqual([]);
  });

  test("tools/call names the tool; the handshake names itself", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    await initialize(KEY_GRANTED);
    const called = await post(KEY_GRANTED, {
      jsonrpc: "2.0",
      id: 3,
      method: "tools/call",
      params: { name: "alpha__echo", arguments: { text: "hi" } },
    });
    expect(called.status).toBe(200);

    const line = await accessLine(KEY_GRANTED, "tools/call");
    expect(line).toContain('mcp_tool="alpha__echo"');
    // The list counts belong to `tools/list` alone — a tool call never
    // borrows the numbers of the request before it.
    expect(line).not.toContain("tools_total");
    expect(line).not.toContain("tools_returned");

    const handshake = await accessLine(KEY_GRANTED, "initialize");
    expect(handshake).not.toContain("mcp_tool");
  });
});
