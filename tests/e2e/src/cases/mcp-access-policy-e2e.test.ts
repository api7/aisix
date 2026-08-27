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

// E2E: layered MCP access against a real gateway + etcd + two real MCP
// upstreams (official TypeScript SDK servers). Three layers of the same
// allow/deny shape intersect on the allow side and union on the deny side.
//
//   env policy      → allow [alpha__*, beta__*], deny [beta__reverse]
//   team T1 policy  → allow [beta__*]
//   team T2 policy  → allow [*]
//   key mcp_access  → the key's own layer; absent = no constraint
//
// Pinned contract:
//   - a key with no block of its own takes the policy layers unchanged;
//   - a team layer narrows the env layer and can never widen it, and
//     neither can a key layer;
//   - an empty allow list on any layer leaves the key nothing;
//   - a deny on any layer wins over every allow;
//   - with no layer present at all the grant is empty;
//   - tools/list hides what tools/call rejects (one ACL, two checkpoints);
//   - policy edits and deletes propagate through the etcd watch path.

const TEAM1 = "team-mcp-t1";
const TEAM2 = "team-mcp-t2";

const ENV_POLICY_ID = "11111111-1111-1111-1111-11111111aaaa";
const T1_POLICY_ID = "11111111-1111-1111-1111-11111111bbbb";
const T2_POLICY_ID = "11111111-1111-1111-1111-11111111cccc";

const KEY_NO_BLOCK = "sk-mcp-no-block";
const KEY_SCOPED = "sk-mcp-scoped";
const KEY_T1 = "sk-mcp-t1";
const KEY_T1_WIDE = "sk-mcp-t1-wide";
const KEY_T2 = "sk-mcp-t2";
const KEY_NARROW = "sk-mcp-t2-narrow";
const KEY_BLOCKED = "sk-mcp-blocked";
const KEY_STALE_MODE = "sk-mcp-stale-mode";

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

describe("mcp access policy e2e: env, team and key layers intersect", () => {
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
        clientInfo: { name: "mcp-acl-e2e", version: "0.1" },
      },
    });
    await post(token, { jsonrpc: "2.0", method: "notifications/initialized" });
    return init.status;
  };

  /** Sorted namespaced tool names visible to `token`, or an HTTP status. */
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

  const callTool = async (
    token: string,
    name: string,
    text: string,
  ): Promise<{ ok: boolean; text?: string; error?: string }> => {
    await initialize(token);
    const r = await post(token, {
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

  const expectList = async (token: string, names: string[]) => {
    const listed = await listToolNames(token);
    expect(listed.status).toBe(200);
    expect(listed.names).toEqual(names);
  };

  /** True once `token` lists exactly `names` — the propagation probe. */
  const listMatches = async (
    token: string,
    names: string[],
  ): Promise<boolean> => {
    const listed = await listToolNames(token);
    return (
      listed.status === 200 &&
      JSON.stringify(listed.names) === JSON.stringify(names)
    );
  };

  const ENV_GRANT = ["alpha__echo", "alpha__reverse", "beta__echo"];

  const EXPECTED: Array<[string, string[]]> = [
    [KEY_NO_BLOCK, ENV_GRANT],
    [KEY_SCOPED, ["alpha__echo", "alpha__reverse"]],
    [KEY_T1, ["beta__echo"]],
    [KEY_T1_WIDE, ["beta__echo"]],
    [KEY_T2, ENV_GRANT],
    [KEY_NARROW, ["alpha__echo"]],
    [KEY_BLOCKED, []],
    [KEY_STALE_MODE, ["alpha__echo", "alpha__reverse"]],
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

    await seed.update("mcp_policies", ENV_POLICY_ID, {
      scope: "env",
      allow: ["alpha__*", "beta__*"],
      deny: ["beta__reverse"],
    });
    await seed.update("mcp_policies", T1_POLICY_ID, {
      scope: "team",
      scope_ref: TEAM1,
      allow: ["beta__*"],
    });
    await seed.update("mcp_policies", T2_POLICY_ID, {
      scope: "team",
      scope_ref: TEAM2,
      allow: ["*"],
    });

    const keyDoc = (
      plaintext: string,
      extra: Record<string, unknown>,
    ): Record<string, unknown> => ({
      key_hash: sha256(plaintext),
      allowed_models: [],
      ...extra,
    });
    await seed.createApiKey(keyDoc(KEY_NO_BLOCK, {}));
    await seed.createApiKey(
      keyDoc(KEY_SCOPED, {
        mcp_access: { allow: ["alpha__*", "beta__reverse"] },
      }),
    );
    await seed.createApiKey(keyDoc(KEY_T1, { team_id: TEAM1 }));
    await seed.createApiKey(
      keyDoc(KEY_T1_WIDE, { team_id: TEAM1, mcp_access: { allow: ["*"] } }),
    );
    await seed.createApiKey(keyDoc(KEY_T2, { team_id: TEAM2 }));
    await seed.createApiKey(
      keyDoc(KEY_NARROW, {
        team_id: TEAM2,
        mcp_access: { allow: ["alpha__*"], deny: ["alpha__reverse"] },
      }),
    );
    await seed.createApiKey(keyDoc(KEY_BLOCKED, { mcp_access: { allow: [] } }));
    // A document a pre-0.10.0 control plane projected: the layered shape
    // plus the retired `mode` selector this build no longer knows. It must
    // stay an ignored unknown field — an api_key row that failed to load
    // would stop the key authenticating every kind of traffic, not just
    // MCP.
    await seed.createApiKey(
      keyDoc(KEY_STALE_MODE, {
        mcp_access: { mode: "deny", allow: ["alpha__*"] },
      }),
    );

    // Probe EVERY key to its expected steady state: keys are written at
    // higher revisions than servers/policies, but each key's list also
    // depends on its own row having landed — probing only one key would
    // let another key's 401 flake a later assertion.
    await waitConfigPropagation(async () => {
      for (const [token, names] of EXPECTED) {
        if (!(await listMatches(token, names))) return false;
      }
      return true;
    });
  }, 60_000);

  afterAll(async () => {
    await app?.exit();
    await alpha?.close();
    await beta?.close();
  });

  test("a stored `mode` selector leaves the row loadable and unenforced", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    // `mode` would have meant "no MCP access" two releases back; here it
    // is inert — the key authenticates and its own allow intersects the
    // env layer as usual.
    await expectList(KEY_STALE_MODE, ["alpha__echo", "alpha__reverse"]);
    const ok = await callTool(KEY_STALE_MODE, "alpha__echo", "hi");
    expect(ok).toEqual({ ok: true, text: "alpha:hi" });
    const denied = await callTool(KEY_STALE_MODE, "beta__echo", "hi");
    expect(denied.ok).toBe(false);
    expect(denied.error).toContain("not available");
  });

  test("a key with no block of its own takes the env layer unchanged", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    await expectList(KEY_NO_BLOCK, ENV_GRANT);
    const ok = await callTool(KEY_NO_BLOCK, "alpha__echo", "hi");
    expect(ok).toEqual({ ok: true, text: "alpha:hi" });
    // beta__reverse is allowed by `beta__*` but denied one layer up.
    const denied = await callTool(KEY_NO_BLOCK, "beta__reverse", "hi");
    expect(denied.ok).toBe(false);
    expect(denied.error).toContain("not available");
  });

  test("the key layer narrows the env layer", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    // The key asks for alpha__* plus beta__reverse; the env deny keeps
    // beta__reverse out, so only the alpha tools survive.
    await expectList(KEY_SCOPED, ["alpha__echo", "alpha__reverse"]);
    const ok = await callTool(KEY_SCOPED, "alpha__reverse", "hi");
    expect(ok).toEqual({ ok: true, text: "ih" });
    for (const tool of ["beta__echo", "beta__reverse"]) {
      const rejected = await callTool(KEY_SCOPED, tool, "hi");
      expect(rejected.ok).toBe(false);
      expect(rejected.error).toContain("not available");
    }
  });

  test("a team layer narrows the env layer; the env deny survives", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    // T1 allows beta__*; intersected with the env layer and minus the env
    // deny on beta__reverse, one tool is left.
    await expectList(KEY_T1, ["beta__echo"]);
    const ok = await callTool(KEY_T1, "beta__echo", "hi");
    expect(ok).toEqual({ ok: true, text: "beta:hi" });
    const envDenied = await callTool(KEY_T1, "beta__reverse", "hi");
    expect(envDenied.ok).toBe(false);
    expect(envDenied.error).toContain("not available");
    // alpha__* is granted by the env layer but not by T1 — the layers
    // intersect, so it is not reachable.
    const narrowed = await callTool(KEY_T1, "alpha__echo", "hi");
    expect(narrowed.ok).toBe(false);
    expect(narrowed.error).toContain("not available");
  });

  test("a wide-open key layer cannot widen the layers above it", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    // Same team as KEY_T1 but asking for `*`: the result is identical.
    await expectList(KEY_T1_WIDE, ["beta__echo"]);
    const rejected = await callTool(KEY_T1_WIDE, "alpha__echo", "hi");
    expect(rejected.ok).toBe(false);
    expect(rejected.error).toContain("not available");
  });

  test("a team layer of `*` leaves the env layer as the only constraint", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    await expectList(KEY_T2, ENV_GRANT);
    const denied = await callTool(KEY_T2, "beta__reverse", "hi");
    expect(denied.ok).toBe(false);
    expect(denied.error).toContain("not available");
  });

  test("all three layers intersect", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    // env [alpha__*, beta__*] ∩ team T2 [*] ∩ key [alpha__*], minus the
    // key's own deny on alpha__reverse.
    await expectList(KEY_NARROW, ["alpha__echo"]);
    const ok = await callTool(KEY_NARROW, "alpha__echo", "hi");
    expect(ok).toEqual({ ok: true, text: "alpha:hi" });
    for (const tool of ["alpha__reverse", "beta__echo"]) {
      const rejected = await callTool(KEY_NARROW, tool, "hi");
      expect(rejected.ok).toBe(false);
      expect(rejected.error).toContain("not available");
    }
  });

  test("an empty allow list on the key leaves it nothing", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    await expectList(KEY_BLOCKED, []);
    const rejected = await callTool(KEY_BLOCKED, "alpha__echo", "hi");
    expect(rejected.ok).toBe(false);
    expect(rejected.error).toContain("not available");
  });

  test("policy edits and deletes propagate through the watch path", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    // Flip T1 to an empty allow list: its member keys lose everything.
    await seed.update("mcp_policies", T1_POLICY_ID, {
      scope: "team",
      scope_ref: TEAM1,
      allow: [],
    });
    await waitConfigPropagation(() => listMatches(KEY_T1, []));
    const rejected = await callTool(KEY_T1, "beta__echo", "hi");
    expect(rejected.ok).toBe(false);
    expect(rejected.error).toContain("not available");

    // Delete the env policy: its deny stops applying, so the key layer of
    // KEY_SCOPED finally admits beta__reverse — and a key with no layer
    // left anywhere drops to no access rather than to everything.
    await seed.delete("mcp_policies", ENV_POLICY_ID);
    await waitConfigPropagation(() =>
      listMatches(KEY_SCOPED, ["alpha__echo", "alpha__reverse", "beta__reverse"]),
    );
    const restored = await callTool(KEY_SCOPED, "beta__reverse", "hi");
    expect(restored).toEqual({ ok: true, text: "ih" });
    await expectList(KEY_NO_BLOCK, []);
  });
});
