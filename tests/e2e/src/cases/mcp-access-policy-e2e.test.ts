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

// E2E: layered MCP access policies against a real gateway + etcd + two real
// MCP upstreams (official TypeScript SDK servers).
//
//   env default policy    → selected [alpha__*], deny [beta__reverse]
//   team T1 policy        → selected [beta__*]      (replaces the env grant)
//   team T2 policy        → all                     (replaces the env grant)
//   per-key mcp_access    → inherit / restrict / deny; absent = legacy
//
// Pinned contract, per key:
//   - legacy keys keep their allowed_tools allow side (no policy widening),
//     but policy deny patterns still subtract;
//   - inherit keys take the team policy when one exists, else the env
//     default; an env-level deny survives a team policy takeover;
//   - restrict intersects (narrow-only); deny grants nothing;
//   - tools/list hides what tools/call rejects (one ACL, two checkpoints);
//   - policy edits and deletes propagate through the etcd watch path.

const TEAM1 = "team-mcp-t1";
const TEAM2 = "team-mcp-t2";

const ENV_POLICY_ID = "11111111-1111-1111-1111-11111111aaaa";
const T1_POLICY_ID = "11111111-1111-1111-1111-11111111bbbb";
const T2_POLICY_ID = "11111111-1111-1111-1111-11111111cccc";

const KEY_LEGACY_SCOPED = "sk-mcp-legacy-scoped";
const KEY_LEGACY_WILD = "sk-mcp-legacy-wild";
const KEY_INHERIT = "sk-mcp-inherit";
const KEY_T1 = "sk-mcp-t1-inherit";
const KEY_T2 = "sk-mcp-t2-inherit";
const KEY_RESTRICT = "sk-mcp-t2-restrict";
const KEY_DENY = "sk-mcp-deny";

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

describe("mcp access policy e2e: env default + team entitlement + key narrowing", () => {
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

  const EXPECTED: Array<[string, string[]]> = [
    [KEY_LEGACY_SCOPED, ["alpha__echo", "alpha__reverse"]],
    [KEY_LEGACY_WILD, ["alpha__echo", "alpha__reverse", "beta__echo"]],
    [KEY_INHERIT, ["alpha__echo", "alpha__reverse"]],
    [KEY_T1, ["beta__echo"]],
    [KEY_T2, ["alpha__echo", "alpha__reverse", "beta__echo"]],
    [KEY_RESTRICT, ["alpha__echo"]],
    [KEY_DENY, []],
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
      mode: "selected",
      allow: ["alpha__*"],
      deny: ["beta__reverse"],
    });
    await seed.update("mcp_policies", T1_POLICY_ID, {
      scope: "team",
      scope_ref: TEAM1,
      mode: "selected",
      allow: ["beta__*"],
    });
    await seed.update("mcp_policies", T2_POLICY_ID, {
      scope: "team",
      scope_ref: TEAM2,
      mode: "all",
    });

    const keyDoc = (
      plaintext: string,
      extra: Record<string, unknown>,
    ): Record<string, unknown> => ({
      key_hash: sha256(plaintext),
      allowed_models: [],
      ...extra,
    });
    await seed.createApiKey(
      keyDoc(KEY_LEGACY_SCOPED, { allowed_tools: ["alpha__*"] }),
    );
    await seed.createApiKey(keyDoc(KEY_LEGACY_WILD, { allowed_tools: ["*"] }));
    await seed.createApiKey(
      keyDoc(KEY_INHERIT, { mcp_access: { mode: "inherit" } }),
    );
    await seed.createApiKey(
      keyDoc(KEY_T1, { team_id: TEAM1, mcp_access: { mode: "inherit" } }),
    );
    await seed.createApiKey(
      keyDoc(KEY_T2, { team_id: TEAM2, mcp_access: { mode: "inherit" } }),
    );
    await seed.createApiKey(
      keyDoc(KEY_RESTRICT, {
        team_id: TEAM2,
        mcp_access: {
          mode: "restrict",
          allow: ["alpha__*"],
          deny: ["alpha__reverse"],
        },
      }),
    );
    await seed.createApiKey(
      keyDoc(KEY_DENY, { allowed_tools: ["*"], mcp_access: { mode: "deny" } }),
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

  test("legacy key keeps its allowed_tools scope", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    await expectList(KEY_LEGACY_SCOPED, ["alpha__echo", "alpha__reverse"]);
    const ok = await callTool(KEY_LEGACY_SCOPED, "alpha__echo", "hi");
    expect(ok).toEqual({ ok: true, text: "alpha:hi" });
    const rejected = await callTool(KEY_LEGACY_SCOPED, "beta__echo", "hi");
    expect(rejected.ok).toBe(false);
    expect(rejected.error).toContain("not available");
  });

  test("policy deny subtracts from a legacy wildcard key without widening it", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    // beta__reverse is hidden by the env deny even though allowed_tools=["*"].
    await expectList(KEY_LEGACY_WILD, [
      "alpha__echo",
      "alpha__reverse",
      "beta__echo",
    ]);
    const denied = await callTool(KEY_LEGACY_WILD, "beta__reverse", "hi");
    expect(denied.ok).toBe(false);
    expect(denied.error).toContain("not available");
    const ok = await callTool(KEY_LEGACY_WILD, "beta__echo", "hi");
    expect(ok).toEqual({ ok: true, text: "beta:hi" });
  });

  test("inherit takes the env default when the key has no team", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    await expectList(KEY_INHERIT, ["alpha__echo", "alpha__reverse"]);
    const ok = await callTool(KEY_INHERIT, "alpha__reverse", "hi");
    expect(ok).toEqual({ ok: true, text: "ih" });
    const rejected = await callTool(KEY_INHERIT, "beta__echo", "hi");
    expect(rejected.ok).toBe(false);
    expect(rejected.error).toContain("not available");
  });

  test("a team policy replaces the env grant; the env deny survives", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    // T1 grants beta__*; the env deny on beta__reverse still subtracts.
    await expectList(KEY_T1, ["beta__echo"]);
    const ok = await callTool(KEY_T1, "beta__echo", "hi");
    expect(ok).toEqual({ ok: true, text: "beta:hi" });
    const envDenied = await callTool(KEY_T1, "beta__reverse", "hi");
    expect(envDenied.ok).toBe(false);
    expect(envDenied.error).toContain("not available");
    // The env grant (alpha__*) is replaced, not unioned.
    const replaced = await callTool(KEY_T1, "alpha__echo", "hi");
    expect(replaced.ok).toBe(false);
    expect(replaced.error).toContain("not available");
  });

  test("a team `all` grant covers both servers, still minus the env deny", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    await expectList(KEY_T2, ["alpha__echo", "alpha__reverse", "beta__echo"]);
    const denied = await callTool(KEY_T2, "beta__reverse", "hi");
    expect(denied.ok).toBe(false);
    expect(denied.error).toContain("not available");
  });

  test("restrict narrows the inherited grant and never widens it", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    // Base is team T2 `all`; the key narrows to alpha__* minus alpha__reverse.
    await expectList(KEY_RESTRICT, ["alpha__echo"]);
    const ok = await callTool(KEY_RESTRICT, "alpha__echo", "hi");
    expect(ok).toEqual({ ok: true, text: "alpha:hi" });
    for (const tool of ["alpha__reverse", "beta__echo"]) {
      const rejected = await callTool(KEY_RESTRICT, tool, "hi");
      expect(rejected.ok).toBe(false);
      expect(rejected.error).toContain("not available");
    }
  });

  test("deny mode grants nothing, whatever allowed_tools says", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    await expectList(KEY_DENY, []);
    const rejected = await callTool(KEY_DENY, "alpha__echo", "hi");
    expect(rejected.ok).toBe(false);
    expect(rejected.error).toContain("not available");
  });

  test("policy edits and deletes propagate through the watch path", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    // Flip T1 to `none`: its member key loses everything.
    await seed.update("mcp_policies", T1_POLICY_ID, {
      scope: "team",
      scope_ref: TEAM1,
      mode: "none",
    });
    await waitConfigPropagation(() => listMatches(KEY_T1, []));
    const rejected = await callTool(KEY_T1, "beta__echo", "hi");
    expect(rejected.ok).toBe(false);
    expect(rejected.error).toContain("not available");

    // Delete the env policy: its deny stops applying to the legacy
    // wildcard key (all four tools return), and an inherit key with no
    // policy left anywhere falls back to no access.
    await seed.delete("mcp_policies", ENV_POLICY_ID);
    await waitConfigPropagation(() =>
      listMatches(KEY_LEGACY_WILD, [
        "alpha__echo",
        "alpha__reverse",
        "beta__echo",
        "beta__reverse",
      ]),
    );
    const restored = await callTool(KEY_LEGACY_WILD, "beta__reverse", "hi");
    expect(restored).toEqual({ ok: true, text: "ih" });
    await expectList(KEY_INHERIT, []);
  });
});
