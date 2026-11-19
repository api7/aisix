import { createHash, randomUUID } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  spawnApp,
  startMcpUpstream,
  waitConfigPropagation,
  waitForLogLine,
  type McpUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// One upstream MCP message — a JSON response body or a single SSE event — is
// capped at 64 MiB. Past the cap the tools/call fails promptly with an error
// naming the cap instead of being buffered whole; a 20 MiB result, larger
// than the MCP client library's own 16 MiB default, still goes through.

const CALLER = "sk-mcp-message-cap";
const MIB = 1024 * 1024;
const CAP = 64 * MIB;
const hash = (s: string) => createHash("sha256").update(s).digest("hex");

describe("upstream MCP message size cap", () => {
  let etcdReachable = false;
  let app: SpawnedApp | undefined;
  let upstream: McpUpstream | undefined;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    upstream = await startMcpUpstream("big", {
      toolFaults: {
        json_over: { kind: "large", textBytes: CAP, framing: "json" },
        sse_over: { kind: "large", textBytes: CAP, framing: "sse" },
        json_20m: { kind: "large", textBytes: 20 * MIB, framing: "json" },
        sse_20m: { kind: "large", textBytes: 20 * MIB, framing: "sse" },
      },
    });
    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.update("mcp_servers", randomUUID(), {
      name: "big",
      url: upstream.url,
      enabled: true,
      timeout_ms: 60_000,
    });
    await seed.createApiKey({
      key_hash: hash(CALLER),
      allowed_models: ["*"],
      mcp_access: { allow: ["*"] },
    });
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${CALLER}` },
      });
      await res.arrayBuffer();
      return res.status === 200;
    });
  }, 120_000);

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  async function callTool(name: string) {
    const res = await fetch(`${app!.proxyUrl}/mcp`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER}`,
        "content-type": "application/json",
        accept: "application/json, text/event-stream",
      },
      body: JSON.stringify({
        jsonrpc: "2.0",
        id: 1,
        method: "tools/call",
        params: { name, arguments: {} },
      }),
    });
    return {
      status: res.status,
      body: (await res.json()) as {
        result?: { content?: Array<{ text?: string }> };
        error?: { code: number; message: string };
      },
    };
  }

  // The JSON overflow is named on the gateway's own failure line. An SSE
  // overflow ends the event stream inside the MCP client, which logs the cap
  // itself and hands the call only a closed transport.
  test.for([
    ["a JSON response", "json_over", "upstream tools/call failed", `exceeded the maximum size of ${CAP} bytes`],
    ["an SSE event", "sse_over", "event stream terminated", `EventTooLarge { max_size: ${CAP} }`],
  ] as const)("%s over the cap fails with an error naming it", { timeout: 90_000 }, async ([, tool, logLine, logText], ctx) => {
    if (!etcdReachable) return ctx.skip();
    const started = Date.now();
    const answer = await callTool(`big__${tool}`);
    expect(answer.status).toBe(200);
    expect(answer.body.result).toBeUndefined();
    expect(answer.body.error?.code).toBe(-32603);
    expect(answer.body.error?.message).toBe("upstream MCP server 'big' failed to call tool");
    await waitForLogLine(
      app!,
      (line) =>
        line.includes("proxy request completed") &&
        line.includes(`mcp_tool="big__${tool}"`) &&
        line.includes('error_class="transport"'),
      `access log for ${tool}`,
    );
    expect(Date.now() - started, "fails on the cap, not on the 60s timeout").toBeLessThan(30_000);
    await waitForLogLine(
      app!,
      (line) => line.includes(logLine) && line.includes(logText),
      `cap error for ${tool}`,
    );
  });

  test.for([
    ["a JSON response", "json_20m"],
    ["an SSE event", "sse_20m"],
  ] as const)("%s of 20 MiB goes through", { timeout: 90_000 }, async ([, tool], ctx) => {
    if (!etcdReachable) return ctx.skip();
    const answer = await callTool(`big__${tool}`);
    expect(answer.status).toBe(200);
    expect(answer.body.error).toBeUndefined();
    expect(answer.body.result?.content?.[0]?.text?.length).toBe(20 * MIB);
  });
});
