import { createHash, randomUUID } from "node:crypto";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
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

// E2E: a key or a policy may point at an MCP server by resource id
// instead of by name — `mcp_access.allow_ids` / `deny_ids`,
// `mcp_policies.allow_ids` / `deny_ids`, and `mcp_rate_limits_by_id`.
// The id form is authoritative for the side it shadows, resolves to
// whatever name the server currently carries, and matches nothing for an
// id that names no server.
//
// Driven against a real gateway + etcd + two real MCP upstreams (official
// TypeScript SDK servers), through the public `/mcp` surface only.

const execFileP = promisify(execFile);

const sha256 = (s: string) => createHash("sha256").update(s).digest("hex");

const KEYS = {
  byId: "sk-mcp-ids-by-id",
  wildcardTool: "sk-mcp-ids-wildcard-tool",
  conflict: "sk-mcp-ids-conflict",
  emptyIds: "sk-mcp-ids-empty",
  partial: "sk-mcp-ids-partial",
  unresolvedOnly: "sk-mcp-ids-unresolved-only",
  denyIds: "sk-mcp-ids-deny",
  envPolicy: "sk-mcp-ids-env-policy",
  rename: "sk-mcp-ids-rename",
  capped: "sk-mcp-ids-capped",
  cappedRename: "sk-mcp-ids-capped-rename",
  sentinel: "sk-mcp-ids-sentinel",
};

const CAP_RPM = 2;

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

describe("mcp server reference ids: grants follow the server id, not its name", () => {
  let app: SpawnedApp | undefined;
  let alpha: McpUpstream | undefined;
  let beta: McpUpstream | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;
  let alphaId = "";
  let renameId = "";
  let renameUrl = "";

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
        clientInfo: { name: "mcp-server-ids-e2e", version: "0.1" },
      },
    });
    await post(token, { jsonrpc: "2.0", method: "notifications/initialized" });
    return init.status;
  };

  const listToolNames = async (token: string): Promise<string[]> => {
    const status = await initialize(token);
    if (status !== 200) return [];
    const r = await post(token, {
      jsonrpc: "2.0",
      id: 2,
      method: "tools/list",
      params: {},
    });
    return (r.json?.result?.tools ?? []).map((t) => t.name).sort();
  };

  /**
   * One `tools/call`, reported as the HTTP status plus the tool text on
   * success. A call the ACL refuses answers 200 with a JSON-RPC error, so
   * `denied` reads the error rather than the status.
   */
  const callTool = async (
    token: string,
    name: string,
  ): Promise<{ status: number; text?: string; error?: string }> => {
    const r = await post(token, {
      jsonrpc: "2.0",
      id: 3,
      method: "tools/call",
      params: { name, arguments: { text: "hi" } },
    });
    return {
      status: r.status,
      text: r.json?.result?.content?.[0]?.text,
      error: r.json?.error?.message,
    };
  };

  /** True when the call reached the tool and returned its answer. */
  const served = async (token: string, name: string): Promise<boolean> => {
    const call = await callTool(token, name);
    return call.status === 200 && call.text !== undefined;
  };

  /**
   * True when the ACL refused the call by name — the gateway's own
   * refusal, not a transport failure and not the upstream erroring.
   */
  const refused = async (token: string, name: string): Promise<boolean> => {
    const call = await callTool(token, name);
    return call.status === 200 && (call.error ?? "").includes("not available");
  };

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    alpha = await startMcpUpstream("alpha");
    beta = await startMcpUpstream("beta");
    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);

    alphaId = randomUUID();
    const betaId = randomUUID();
    renameId = randomUUID();
    renameUrl = beta.url;

    await seed.update("mcp_servers", alphaId, {
      name: "alpha",
      url: alpha.url,
      enabled: true,
    });
    await seed.update("mcp_servers", betaId, {
      name: "beta",
      url: beta.url,
      enabled: true,
    });
    // Its own server, so the rename case can move a name without
    // disturbing the inventory every other case asserts on.
    await seed.update("mcp_servers", renameId, {
      name: "gamma",
      url: renameUrl,
      enabled: true,
    });

    const key = (
      plaintext: string,
      extra: Record<string, unknown>,
    ): Record<string, unknown> => ({
      key_hash: sha256(plaintext),
      allowed_models: [],
      ...extra,
    });

    // Grant by id, tool by name.
    await seed.createApiKey(
      key(KEYS.byId, {
        mcp_access: {
          allow: [],
          allow_ids: [{ server_id: alphaId, tool: "echo" }],
        },
      }),
    );
    // A `*` tool covers the whole server the id names.
    await seed.createApiKey(
      key(KEYS.wildcardTool, {
        mcp_access: {
          allow: [],
          allow_ids: [{ server_id: alphaId, tool: "*" }],
        },
      }),
    );
    // Both spellings present and disagreeing: the id side decides.
    await seed.createApiKey(
      key(KEYS.conflict, {
        mcp_access: {
          allow: ["beta__*"],
          allow_ids: [{ server_id: alphaId, tool: "*" }],
        },
      }),
    );
    // An empty id array is authoritative too — it grants nothing, and the
    // wide-open name side is not read.
    await seed.createApiKey(
      key(KEYS.emptyIds, { mcp_access: { allow: ["*"], allow_ids: [] } }),
    );
    // One unresolvable id beside a good one.
    await seed.createApiKey(
      key(KEYS.partial, {
        mcp_access: {
          allow: [],
          allow_ids: [
            { server_id: randomUUID(), tool: "*" },
            { server_id: alphaId, tool: "echo" },
          ],
        },
      }),
    );
    // Nothing but an unresolvable id, with a wide-open name side that must
    // stay unread.
    await seed.createApiKey(
      key(KEYS.unresolvedOnly, {
        mcp_access: {
          allow: ["*"],
          allow_ids: [{ server_id: randomUUID(), tool: "*" }],
        },
      }),
    );
    // Deny by id subtracts from a wide-open allow side.
    await seed.createApiKey(
      key(KEYS.denyIds, {
        mcp_access: {
          allow: ["*"],
          deny_ids: [{ server_id: alphaId, tool: "reverse" }],
        },
      }),
    );
    // The rename case: the grant names `gamma`'s id, never its name.
    await seed.createApiKey(
      key(KEYS.rename, {
        mcp_access: {
          allow: [],
          allow_ids: [{ server_id: renameId, tool: "*" }],
        },
      }),
    );
    // Per-server limits keyed by id, one before and one across a rename.
    await seed.createApiKey(
      key(KEYS.capped, {
        mcp_access: { allow: ["*"] },
        mcp_rate_limits: { beta: { rpm: CAP_RPM } },
        mcp_rate_limits_by_id: { [alphaId]: { rpm: CAP_RPM } },
      }),
    );
    await seed.createApiKey(
      key(KEYS.cappedRename, {
        mcp_access: { allow: ["*"] },
        mcp_rate_limits_by_id: { [renameId]: { rpm: CAP_RPM } },
      }),
    );
    // The environment MCP access policy, written by id. Its own key, and
    // an `mcp_policies` row rather than a key field, because a policy row
    // travels a different path from etcd to the ACL than a key's own
    // `mcp_access` block does and only this layer proves it arrives.
    await seed.createApiKey(
      key(KEYS.envPolicy, {
        team_id: "team-mcp-ids",
        mcp_access: { allow: ["*"] },
      }),
    );
    await seed.update("mcp_policies", randomUUID(), {
      scope: "team",
      scope_ref: "team-mcp-ids",
      allow: [],
      allow_ids: [{ server_id: alphaId, tool: "echo" }],
      enabled: true,
    });

    // Seeded last and unrestricted: gating on it implies the whole seed
    // set has landed without exercising any behavior under test.
    await seed.createApiKey(
      key(KEYS.sentinel, { mcp_access: { allow: ["*"] } }),
    );

    // `tools/list` is unmetered, so this probe spends no key's budget.
    await waitConfigPropagation(async () => {
      const names = await listToolNames(KEYS.sentinel);
      return (
        names.length === 6 &&
        names.includes("alpha__echo") &&
        names.includes("beta__echo") &&
        names.includes("gamma__echo")
      );
    });
  }, 60_000);

  afterAll(async () => {
    await app?.exit();
    await alpha?.close();
    await beta?.close();
  });

  test("an id grant covers exactly the named server's named tool", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    expect(await listToolNames(KEYS.byId)).toEqual(["alpha__echo"]);
    expect(await served(KEYS.byId, "alpha__echo")).toBe(true);
    expect(await refused(KEYS.byId, "alpha__reverse")).toBe(true);
    expect(await refused(KEYS.byId, "beta__echo")).toBe(true);
  });

  test("a `*` tool on an id grant covers that server and no other", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    expect(await listToolNames(KEYS.wildcardTool)).toEqual([
      "alpha__echo",
      "alpha__reverse",
    ]);
    expect(await refused(KEYS.wildcardTool, "beta__echo")).toBe(true);
  });

  test("the id side decides when both spellings are present", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    // `allow` says beta, `allow_ids` says alpha. The names are not read.
    expect(await listToolNames(KEYS.conflict)).toEqual([
      "alpha__echo",
      "alpha__reverse",
    ]);
    expect(await refused(KEYS.conflict, "beta__echo")).toBe(true);

    // An empty id array is the authoritative "nothing", even beside `*`.
    expect(await listToolNames(KEYS.emptyIds)).toEqual([]);
    expect(await refused(KEYS.emptyIds, "alpha__echo")).toBe(true);
  });

  test("an unresolvable server id matches nothing and spares its neighbours", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    expect(await listToolNames(KEYS.partial)).toEqual(["alpha__echo"]);
    expect(await served(KEYS.partial, "alpha__echo")).toBe(true);

    // The whole grant is unresolvable, and the wide-open name side stays
    // unread rather than filling in for it.
    expect(await listToolNames(KEYS.unresolvedOnly)).toEqual([]);
    expect(await refused(KEYS.unresolvedOnly, "alpha__echo")).toBe(true);
  });

  test("a deny written by id subtracts from an allow written by name", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    const names = await listToolNames(KEYS.denyIds);
    expect(names).toContain("alpha__echo");
    expect(names).toContain("beta__reverse");
    expect(names).not.toContain("alpha__reverse");
    expect(await refused(KEYS.denyIds, "alpha__reverse")).toBe(true);
    expect(await served(KEYS.denyIds, "beta__reverse")).toBe(true);
  });

  test("a policy layer may write its allow side by id too", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    // The key's own layer is wide open, so everything below is the team
    // policy's id-spelled allow side narrowing it.
    expect(await listToolNames(KEYS.envPolicy)).toEqual(["alpha__echo"]);
    expect(await served(KEYS.envPolicy, "alpha__echo")).toBe(true);
    expect(await refused(KEYS.envPolicy, "alpha__reverse")).toBe(true);
    expect(await refused(KEYS.envPolicy, "beta__echo")).toBe(true);

    // A key outside that team is not narrowed by it, so the policy really
    // is what produced the result above.
    expect(await served(KEYS.sentinel, "beta__echo")).toBe(true);
  });

  test("a per-server limit keyed by id binds on that server alone", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    // The limiter buckets on fixed wall-clock minutes; keep the burst
    // inside one window so the 429 cannot straddle a roll-over.
    await awaitWindowHeadroom();

    for (let i = 0; i < CAP_RPM; i++) {
      expect((await callTool(KEYS.capped, "alpha__echo")).status).toBe(200);
    }
    expect((await callTool(KEYS.capped, "alpha__echo")).status).toBe(429);
    // The same server's other tool shares that server's counter.
    expect((await callTool(KEYS.capped, "alpha__reverse")).status).toBe(429);
    // `beta` is capped only by the NAME form, which the id form shadows —
    // so this key has no limit on beta at all.
    for (let i = 0; i < CAP_RPM + 2; i++) {
      expect((await callTool(KEYS.capped, "beta__echo")).status).toBe(200);
    }
  }, 60_000);

  // Last: the rename mutates shared state, and the server it renames is
  // used by no other case.
  test("renaming a server moves the grant and the limit with it, documents untouched", async (ctx) => {
    if (!etcdReachable || !app || !seed) return ctx.skip();

    expect(await served(KEYS.rename, "gamma__echo")).toBe(true);

    // Same etcd key (same resource id), new name. Neither key document is
    // rewritten anywhere in this test.
    await seed.update("mcp_servers", renameId, {
      name: "gamma-v2",
      url: renameUrl,
      enabled: true,
    });
    // Gate on the rename landing, observed through the unrestricted
    // sentinel key rather than through the keys under test.
    await waitConfigPropagation(async () => {
      const names = await listToolNames(KEYS.sentinel);
      return (
        names.includes("gamma-v2__echo") && !names.includes("gamma__echo")
      );
    });

    // The grant followed the id into the server's new namespace...
    expect(await listToolNames(KEYS.rename)).toEqual([
      "gamma-v2__echo",
      "gamma-v2__reverse",
    ]);
    expect(await served(KEYS.rename, "gamma-v2__echo")).toBe(true);
    // ...and did not linger on the old name, which now names no server.
    expect(await served(KEYS.rename, "gamma__echo")).toBe(false);

    // So did the per-server limit.
    await awaitWindowHeadroom();
    for (let i = 0; i < CAP_RPM; i++) {
      expect(
        (await callTool(KEYS.cappedRename, "gamma-v2__echo")).status,
      ).toBe(200);
    }
    expect((await callTool(KEYS.cappedRename, "gamma-v2__echo")).status).toBe(
      429,
    );
  }, 60_000);
});

// The id spelling is a control-plane projection: a resources file derives
// its ids from its own entry names, so an id written there could never
// resolve. `aisix validate` refuses each of them by name rather than
// loading a file whose grants and limits silently point at nothing.
describe("resources file: every MCP server reference id spelling is refused", () => {
  const BIN_PATH =
    process.env.AISIX_BIN ??
    join(process.cwd(), "..", "..", "target", "debug", "aisix");

  const PRELUDE = [
    '_format_version: "1"',
    "mcp_servers:",
    "  - name: github",
    "    url: https://example.test/mcp",
    "api_keys:",
    "  - display_name: k",
    "    key_env: MCP_REF_IDS_CALLER_KEY",
    "    allowed_models: []",
  ].join("\n");

  const ID = "11111111-1111-1111-1111-111111111111";
  const ENV = { ...process.env, MCP_REF_IDS_CALLER_KEY: "sk-caller" };

  const CASES = [
    {
      label: "per-server rate limits",
      field: "mcp_rate_limits_by_id",
      withId: `\n    mcp_rate_limits_by_id:\n      "${ID}":\n        rpm: 1\n`,
      without: "",
    },
    {
      label: "mcp_access allow",
      field: "mcp_access.allow_ids",
      withId: `\n    mcp_access:\n      allow: []\n      allow_ids:\n        - server_id: "${ID}"\n          tool: "*"\n`,
      without: `\n    mcp_access:\n      allow: []\n`,
    },
    {
      label: "mcp_access deny",
      field: "mcp_access.deny_ids",
      withId: `\n    mcp_access:\n      allow: ["*"]\n      deny_ids:\n        - server_id: "${ID}"\n          tool: drop\n`,
      without: `\n    mcp_access:\n      allow: ["*"]\n`,
    },
  ];

  test("each id field is rejected by name, and the same file without it validates", async () => {
    const dir = await mkdtemp(join(tmpdir(), "aisix-mcp-ref-ids-"));
    try {
      for (const { label, field, withId, without } of CASES) {
        const bad = join(dir, `bad-${label.replace(/\W+/g, "-")}.yaml`);
        await writeFile(bad, PRELUDE + withId, "utf8");
        let failure: (Error & { code?: number; stderr?: string }) | undefined;
        try {
          await execFileP(BIN_PATH, ["validate", "--resources", bad], { env: ENV });
        } catch (e) {
          failure = e as Error & { code?: number; stderr?: string };
        }
        if (!failure) {
          throw new Error(`${label}: expected \`aisix validate\` to fail`);
        }
        expect(failure.code, label).toBe(1);
        expect(String(failure.stderr), label).toContain(
          `does not accept \`${field}\``,
        );

        // The same file with the id field removed validates, so the
        // refusal is that field and not anything else in the fixture.
        const good = join(dir, `good-${label.replace(/\W+/g, "-")}.yaml`);
        await writeFile(good, PRELUDE + without, "utf8");
        const ok = await execFileP(BIN_PATH, ["validate", "--resources", good], {
          env: ENV,
        });
        expect(ok.stdout, label).toContain("OK:");
      }
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  }, 60_000);
});
