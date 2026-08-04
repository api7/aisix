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

// E2E: a credentialed MCP upstream reached over cleartext http logs one
// warning — and exactly one, deduped across requests, because the gateway
// is rebuilt from the snapshot per request. The warning never rejects:
// plain-http upstreams inside a private network are a lawful deployment,
// so calls keep working (#879).

const KEY = "sk-cleartext-warn-e2e";
const sha256 = (s: string) => createHash("sha256").update(s).digest("hex");
const WARN_MARK = "sent over cleartext http";

describe("mcp cleartext credential warning e2e", () => {
  let app: SpawnedApp | undefined;
  let upstream: McpUpstream | undefined;
  let etcdReachable = false;

  const initialize = async (): Promise<number> => {
    const res = await fetch(`${app!.proxyUrl}/mcp`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${KEY}`,
        "content-type": "application/json",
        accept: "application/json, text/event-stream",
      },
      body: JSON.stringify({
        jsonrpc: "2.0",
        id: 1,
        method: "initialize",
        params: {
          protocolVersion: "2025-11-25",
          capabilities: {},
          clientInfo: { name: "cleartext-warn-e2e", version: "0.1" },
        },
      }),
    });
    return res.status;
  };

  const warnCount = (): number =>
    app!.output().split(WARN_MARK).length - 1;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startMcpUpstream("alpha");
    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);

    // Credentialed (bearer) server over plain http — the harness upstream
    // ignores auth headers, so calls succeed while the credential is the
    // warning's subject.
    await seed.update("mcp_servers", randomUUID(), {
      display_name: "alpha",
      url: upstream.url,
      enabled: true,
      auth_type: "bearer",
      secret: "warn-e2e-token",
    });
    await seed.createApiKey({
      key_hash: sha256(KEY),
      allowed_models: [],
      allowed_tools: ["*"],
    });

    await waitConfigPropagation(async () => (await initialize()) === 200);
  }, 60_000);

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("warns once about the cleartext credential and keeps serving", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    // The propagation probe already drove at least one request through the
    // gateway; poll for the line rather than racing the stderr pipe.
    await waitConfigPropagation(async () => warnCount() >= 1);
    expect(warnCount()).toBe(1);
    expect(app.output()).toContain("alpha");

    // …and it stays at one across further requests (per-process dedup),
    // while the endpoint keeps answering — a warning, not a rejection. The
    // short settle lets any (buggy) extra warn line reach the pipe buffer
    // before the count is read, so a dedup regression can actually fail
    // this assertion.
    expect(await initialize()).toBe(200);
    expect(await initialize()).toBe(200);
    await new Promise((resolve) => setTimeout(resolve, 250));
    expect(warnCount()).toBe(1);
  });
});
