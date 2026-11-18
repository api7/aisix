import { execFile } from "node:child_process";
import { createHash, randomUUID } from "node:crypto";
import { join } from "node:path";
import { promisify } from "node:util";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import { parse as parseYaml } from "yaml";
import {
  EtcdClient,
  etcdEndpoint,
  ProxyClient,
  SeedClient,
  spawnApp,
  startMcpUpstream,
  waitConfigPropagation,
  type McpUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// Anonymous MCP access must survive `aisix export` into a resources file:
// the file-mode gateway serves an uncredentialed caller as the same bound
// API key the etcd-backed gateway does. The control plane writes that key
// as `anonymous.api_key_id`, a resource id, and the export carries the
// settings row and the key under the ids the control plane gave them.

const execFileP = promisify(execFile);
const BIN_PATH =
  process.env.AISIX_BIN ?? join(process.cwd(), "..", "..", "target", "debug", "aisix");

const sha256 = (s: string) => createHash("sha256").update(s).digest("hex");
const PRINCIPAL_SECRET = `sk-export-anon-${randomUUID()}`;
const SENTINEL = `sk-export-anon-sentinel-${randomUUID()}`;

describe("aisix export: anonymous MCP access bound to an api key", () => {
  let etcdApp: SpawnedApp | undefined;
  let fileApp: SpawnedApp | undefined;
  let docs: McpUpstream | undefined;
  let exported = "";
  let exportStderr = "";
  let principalId = "";

  const post = async (app: SpawnedApp, path: string, body: unknown) => {
    const res = await fetch(`${app.proxyUrl}${path}`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        accept: "application/json, text/event-stream",
      },
      body: JSON.stringify(body),
    });
    const text = await res.text();
    let json;
    try {
      json = text ? JSON.parse(text) : undefined;
    } catch {
      json = undefined;
    }
    return { status: res.status, json };
  };

  /** Anonymous `initialize` + `tools/list` on `path`; names on success. */
  const anonymousTools = async (app: SpawnedApp, path: string) => {
    const init = await post(app, path, {
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: {
        protocolVersion: "2025-11-25",
        capabilities: {},
        clientInfo: { name: "export-anon-e2e", version: "0.1" },
      },
    });
    if (init.status !== 200) return { status: init.status };
    await post(app, path, { jsonrpc: "2.0", method: "notifications/initialized" });
    const list = await post(app, path, {
      jsonrpc: "2.0",
      id: 2,
      method: "tools/list",
      params: {},
    });
    const tools: Array<{ name: string }> | undefined = list.json?.result?.tools;
    return { status: list.status, names: tools?.map((t) => t.name).sort() };
  };

  beforeAll(async () => {
    const etcd = new EtcdClient();
    expect(await etcd.ping(), "this e2e requires etcd").toBe(true);
    docs = await startMcpUpstream("docs");
    etcdApp = await spawnApp();
    const seed = new SeedClient(etcd, etcdApp.etcdPrefix);

    const docsId = randomUUID();
    await seed.update("mcp_servers", docsId, { name: "docs", url: docs.url, enabled: true });
    const principal = await seed.createApiKey({
      key_hash: sha256(PRINCIPAL_SECRET),
      allowed_models: [],
      mcp_access: { allow: ["docs__*"] },
    });
    principalId = principal.id;
    // Both allowlist spellings, the way the control plane projects them.
    await seed.update("mcp_auth_settings", randomUUID(), {
      anonymous: {
        api_key_id: principal.id,
        source_cidrs: ["127.0.0.0/8", "::1/128"],
        servers: ["docs"],
        server_ids: [docsId],
      },
    });
    // Seeded after the settings row: watch events apply in revision
    // order, so this key authenticating means the row has landed too.
    await seed.createApiKey({ key_hash: sha256(SENTINEL), allowed_models: [] });
    const gate = new ProxyClient(etcdApp.proxyUrl, SENTINEL);
    await waitConfigPropagation(async () => (await gate.listModels()).status === 200);

    const { stdout, stderr } = await execFileP(
      BIN_PATH,
      ["export", "--etcd", etcdEndpoint(), "--prefix", etcdApp.etcdPrefix, "--reveal-secrets"],
      { timeout: 30_000 },
    );
    exported = stdout;
    exportStderr = stderr;
  }, 60_000);

  afterAll(async () => {
    await fileApp?.exit();
    await etcdApp?.exit();
    await docs?.close();
  });

  test("the exported file serves anonymous callers as the etcd gateway does", async () => {
    expect(exportStderr).not.toContain("anonymous");
    const doc = parseYaml(exported) as {
      api_keys: Array<{ id: string }>;
      mcp_auth_settings: Array<{ anonymous: { api_key_id: string } }>;
    };
    expect(doc.mcp_auth_settings[0]!.anonymous.api_key_id).toBe(principalId);
    expect(doc.api_keys.map((k) => k.id)).toContain(principalId);
    fileApp = await spawnApp({ resourcesFile: exported });
    for (const [label, app] of [["etcd", etcdApp!], ["file", fileApp]] as const) {
      const listed = await anonymousTools(app, "/mcp/docs");
      expect(listed.status, label).toBe(200);
      expect(listed.names).toEqual(["echo", "reverse"]);
    }
  }, 60_000);
});
