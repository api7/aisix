import { createHash, randomUUID } from "node:crypto";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
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

// E2E: the anonymous MCP allowlist may name its servers by resource id —
// `mcp_auth_settings.anonymous.server_ids` — instead of by name. The id
// form is authoritative for the whole allowlist (the `/mcp/{server}` entry
// gate AND the ceiling laid over the bound principal on the aggregated
// endpoint), resolves to whatever name each server currently carries, and
// admits nothing for an id that names no server.
//
// Driven against a real gateway + etcd + three real MCP upstreams, through
// the public `/mcp` surface only. Every case reconfigures the singleton
// settings row and waits for the new configuration to be observable, so
// they run in order.

const execFileP = promisify(execFile);

const sha256 = (s: string) => createHash("sha256").update(s).digest("hex");

const PRINCIPAL_SECRET = `sk-anon-ids-principal-${randomUUID()}`;
const SENTINEL = "sk-anon-ids-sentinel";

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

describe("mcp anonymous access: the allowlist may name servers by id", () => {
  let app: SpawnedApp | undefined;
  let docs: McpUpstream | undefined;
  let kb: McpUpstream | undefined;
  let roam: McpUpstream | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;
  const settingsId = randomUUID();
  let principalId = "";
  const docsId = randomUUID();
  const kbId = randomUUID();
  const roamId = randomUUID();
  let roamUrl = "";

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
        clientInfo: { name: "mcp-anon-ids-e2e", version: "0.1" },
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

  /** True when the call reached the tool and returned its answer. */
  const served = async (
    path: string,
    token: string | undefined,
    name: string,
  ): Promise<boolean> => {
    await initialize(path, token);
    const r = await post(path, token, {
      jsonrpc: "2.0",
      id: 3,
      method: "tools/call",
      params: { name, arguments: { text: "hi" } },
    });
    if (r.status !== 200 || r.json?.error) return false;
    const result = r.json?.result;
    return !!result && !result.isError && result.content?.[0]?.text !== undefined;
  };

  /**
   * Rewrite the singleton settings row's anonymous block and wait for it
   * to reach the gateway.
   *
   * The gate is a FRESH api key seeded after the row and watched until it
   * authenticates, never a request on the anonymous path: watch events
   * apply in revision order, so the key answering implies the row before
   * it landed — and a gate that exercised the behavior under test would
   * fail by timing out instead of by an assertion.
   */
  const applyAnonymous = async (anonymous: Record<string, unknown>) => {
    await seed!.update("mcp_auth_settings", settingsId, {
      anonymous: {
        api_key_id: principalId,
        // The e2e client connects over loopback; both families, since
        // `localhost` may resolve either way.
        source_cidrs: ["127.0.0.0/8", "::1/128"],
        aggregate_entry: true,
        ...anonymous,
      },
    });
    const secret = `sk-anon-ids-gate-${randomUUID()}`;
    await seed!.createApiKey({
      key_hash: sha256(secret),
      allowed_models: [],
    });
    const gate = new ProxyClient(app!.proxyUrl, secret);
    await waitConfigPropagation(async () => (await gate.listModels()).status === 200);
  };

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    docs = await startMcpUpstream("docs");
    kb = await startMcpUpstream("kb");
    roam = await startMcpUpstream("roam");
    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);
    roamUrl = roam.url;

    for (const [id, name, url] of [
      [docsId, "docs", docs.url],
      [kbId, "kb", kb.url],
      [roamId, "roam", roamUrl],
    ] as const) {
      await seed.update("mcp_servers", id, { name, url, enabled: true });
    }

    // The anonymous principal holds a WILDCARD grant on purpose: every
    // narrowing asserted below comes from the anonymous allowlist, never
    // from the key.
    principalId = (
      await seed.createApiKey({
        key_hash: sha256(PRINCIPAL_SECRET),
        allowed_models: [],
        mcp_access: { allow: ["*"] },
      })
    ).id;
    // An authenticated key with the same wildcard grant: it observes the
    // registered inventory without going through the anonymous path, so a
    // propagation gate can watch it instead of the behavior under test.
    await seed.createApiKey({
      key_hash: sha256(SENTINEL),
      allowed_models: [],
      mcp_access: { allow: ["*"] },
    });

    // `servers` deliberately names a DIFFERENT server than `server_ids`:
    // every assertion below distinguishes which spelling was read.
    await applyAnonymous({ servers: ["kb"], server_ids: [docsId] });
  }, 60_000);

  afterAll(async () => {
    await app?.exit();
    await docs?.close();
    await kb?.close();
    await roam?.close();
  });

  test("the id spelling decides the entry gate and the ceiling", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    // `server_ids` names docs; `servers` names kb and is not read.
    const listed = await listToolNames("/mcp/docs", undefined);
    expect(listed.status).toBe(200);
    expect(listed.names).toEqual(["echo", "reverse"]);
    expect(await served("/mcp/docs", undefined, "echo")).toBe(true);

    // kb IS registered and IS named by `servers` — it must answer exactly
    // like a server that does not exist, so an anonymous prober cannot
    // tell "unlisted" from "unknown".
    for (const server of ["kb", "roam", "ghost"]) {
      expect(await initialize(`/mcp/${server}`, undefined), server).toBe(401);
    }

    // The allowlist is a CEILING, not merely an entry gate: the bound
    // principal's own grant is `*`, so anything visible here that is not
    // docs would be the ceiling failing.
    const aggregated = await listToolNames("/mcp", undefined);
    expect(aggregated.status).toBe(200);
    expect(aggregated.names).toEqual(["docs__echo", "docs__reverse"]);
    expect(await served("/mcp", undefined, "kb__echo")).toBe(false);

    // And none of it touches the authenticated path.
    const authed = await listToolNames("/mcp", SENTINEL);
    expect(authed.names).toEqual([
      "docs__echo",
      "docs__reverse",
      "kb__echo",
      "kb__reverse",
      "roam__echo",
      "roam__reverse",
    ]);
  }, 60_000);

  test("renaming an allowlisted server moves anonymous access with it", async (ctx) => {
    if (!etcdReachable || !app || !seed) return ctx.skip();

    await applyAnonymous({ servers: ["docs"], server_ids: [roamId] });
    expect(await initialize("/mcp/roam", undefined)).toBe(200);
    expect(await initialize("/mcp/docs", undefined)).toBe(401);

    // Same etcd key (same resource id), new name. The settings row is NOT
    // rewritten anywhere below.
    await seed.update("mcp_servers", roamId, {
      name: "roam-v2",
      url: roamUrl,
      enabled: true,
    });
    // Gate on the rename landing, observed through the authenticated
    // sentinel rather than through the anonymous path under test.
    await waitConfigPropagation(async () => {
      const names = (await listToolNames("/mcp", SENTINEL)).names ?? [];
      return names.includes("roam-v2__echo") && !names.includes("roam__echo");
    });

    // Anonymous access followed the id into the server's new namespace...
    const listed = await listToolNames("/mcp/roam-v2", undefined);
    expect(listed.status).toBe(200);
    expect(listed.names).toEqual(["echo", "reverse"]);
    expect(await served("/mcp/roam-v2", undefined, "echo")).toBe(true);
    // ...and did not linger on the old name, which now names no server.
    expect(await initialize("/mcp/roam", undefined)).toBe(401);
    // The ceiling moved too, not only the entry gate.
    expect((await listToolNames("/mcp", undefined)).names).toEqual([
      "roam-v2__echo",
      "roam-v2__reverse",
    ]);
  }, 60_000);

  test("an empty server_ids denies every anonymous entry", async (ctx) => {
    if (!etcdReachable || !app || !seed) return ctx.skip();

    // `servers` still names two registered servers; the empty id array is
    // authoritative and admits neither.
    await applyAnonymous({ servers: ["docs", "kb"], server_ids: [] });

    for (const server of ["docs", "kb", "roam-v2"]) {
      expect(await initialize(`/mcp/${server}`, undefined), server).toBe(401);
    }
    // The aggregated entry closes with them, `aggregate_entry: true` and
    // all: an open door onto an empty room would admit an uncredentialed
    // caller as the principal with no tool to reach, and suppress the
    // `WWW-Authenticate` hint a standard client follows to sign in.
    expect(await initialize("/mcp", undefined)).toBe(401);

    // The authenticated path is untouched by all of it.
    expect((await listToolNames("/mcp", SENTINEL)).names?.length).toBe(6);
  }, 60_000);

  test("an unresolvable id is not the same as an empty allowlist", async (ctx) => {
    if (!etcdReachable || !app || !seed) return ctx.skip();

    // A server the gateway cannot resolve is a transient the operator did
    // not ask for, not a deliberate "no server": the aggregated entry keeps
    // its pre-existing behavior there (open, under a ceiling admitting
    // nothing) rather than closing the way an empty allowlist does.
    await applyAnonymous({ servers: ["docs"], server_ids: [randomUUID()] });
    expect(await initialize("/mcp", undefined)).toBe(200);
    expect((await listToolNames("/mcp", undefined)).names).toEqual([]);
    expect(await initialize("/mcp/docs", undefined)).toBe(401);
  }, 60_000);

  test("an id naming no registered server admits nothing", async (ctx) => {
    if (!etcdReachable || !app || !seed) return ctx.skip();

    await applyAnonymous({ servers: ["docs"], server_ids: [randomUUID(), kbId] });
    expect(await initialize("/mcp/kb", undefined)).toBe(200);

    // The unresolvable id contributed nothing and left its neighbour
    // alone; `docs`, named only by the unread name form, stays closed.
    expect((await listToolNames("/mcp", undefined)).names).toEqual([
      "kb__echo",
      "kb__reverse",
    ]);
    expect(await initialize("/mcp/docs", undefined)).toBe(401);
    expect(await served("/mcp", undefined, "docs__echo")).toBe(false);
    expect(await served("/mcp/kb", undefined, "echo")).toBe(true);
  }, 60_000);
});

// Two write-path refusals the gateway binary enforces on its own, with no
// etcd involved: the resources file cannot name an MCP server by the id a
// control plane assigned it, and a registered server's name may not carry
// a `*`.
describe("resources file: MCP server names and the anonymous ceiling", () => {
  const BIN_PATH =
    process.env.AISIX_BIN ??
    join(process.cwd(), "..", "..", "target", "debug", "aisix");

  const run = async (
    contents: string,
  ): Promise<{ ok: boolean; out: string }> => {
    const dir = await mkdtemp(join(tmpdir(), "aisix-anon-ids-"));
    try {
      const path = join(dir, "resources.yaml");
      await writeFile(path, contents, "utf8");
      try {
        const ok = await execFileP(BIN_PATH, ["validate", "--resources", path], {
          env: { ...process.env, ANON_CALLER_KEY: "sk-caller" },
        });
        return { ok: true, out: ok.stdout };
      } catch (e) {
        const err = e as Error & { code?: number; stderr?: string };
        expect(err.code).toBe(1);
        return { ok: false, out: String(err.stderr) };
      }
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  };

  const PRELUDE = [
    '_format_version: "1"',
    "mcp_servers:",
    "  - name: docs",
    "    url: https://example.test/mcp",
    "api_keys:",
    "  - display_name: k",
    "    key_env: ANON_CALLER_KEY",
    "    allowed_models: []",
    "mcp_auth_settings:",
    "  - anonymous:",
    "      api_key_id: k",
    '      source_cidrs: ["10.0.0.0/8"]',
    '      servers: ["docs"]',
    "",
  ].join("\n");

  test("`anonymous.server_ids` is refused, and the same file without it loads", async () => {
    const bad = await run(
      `${PRELUDE}      server_ids: ["11111111-1111-1111-1111-111111111111"]\n`,
    );
    expect(bad.ok).toBe(false);
    expect(bad.out).toContain("does not accept `anonymous.server_ids`");
    expect(bad.out).toContain("servers");

    const good = await run(PRELUDE);
    expect(good.ok).toBe(true);
    expect(good.out).toContain("OK:");
  }, 60_000);

  test("an MCP server name containing `*` is refused", async () => {
    // The name is pasted into the `<server>__*` glob patterns every
    // name-form grant, deny and anonymous ceiling is written as, so `gh*`
    // would reach `ghost`'s tools as well as its own.
    const bad = await run(
      ['_format_version: "1"', "mcp_servers:", '  - name: "gh*"', "    url: https://example.test/mcp", ""].join("\n"),
    );
    expect(bad.ok).toBe(false);
    expect(bad.out).toContain("/name");

    // The same file with the `*` removed loads, so the refusal is that
    // character and not anything else in the fixture.
    const good = await run(
      ['_format_version: "1"', "mcp_servers:", "  - name: gh", "    url: https://example.test/mcp", ""].join("\n"),
    );
    expect(good.ok).toBe(true);
    expect(good.out).toContain("OK:");
  }, 60_000);
});
