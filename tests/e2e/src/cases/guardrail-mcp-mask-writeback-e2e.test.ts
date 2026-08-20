import { createHash, randomUUID } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  decodedTextFor,
  EtcdClient,
  ProxyClient,
  SeedClient,
  spawnApp,
  startMcpUpstream,
  startMockSls,
  waitConfigPropagation,
  waitForToken,
  type McpUpstream,
  type MockSls,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: the MCP mask write-back channel (AISIX-Cloud#1330), against a real
// DP + etcd + a real MCP upstream (official TypeScript SDK server) + the
// SLS mock as the SOC export target.
//
// Two DP instances share ONE upstream:
//   - appG: pii mask guardrail (hook both) + a full-content SLS exporter;
//   - appP: no guardrail, no exporter — the byte-for-byte baseline.
// The baseline lives in its own app so raw sensitive values never reach
// the SOC export legitimately; anything raw in SLS is therefore a leak.
//
// Pinned contract:
//   - request direction: the upstream receives the tool arguments with ONLY
//     the masked spans rewritten (byte-diff against the baseline app's
//     upstream request);
//   - response direction: the client receives the tool result with ONLY the
//     masked spans rewritten — full-body byte-diff — covering a text block,
//     an embedded resource's `resource.text` (the compile-log shape that
//     previously escaped scanning entirely), and `structuredContent` leaves;
//     the body still parses as JSON;
//   - rewrite never blocks: HTTP 200, no JSON-RPC error, no `isError`;
//   - the SOC export carries the POST-MASK content and the detector counts,
//     never the raw values;
//   - Chinese and English label forms both rewrite.

const KEY = "sk-mcp-writeback-e2e";
const sha256 = (s: string) => createHash("sha256").update(s).digest("hex");

const CREDENTIAL_REF = "mock";
const SLS_PROJECT = "aisix-e2e-obs";
const FULL_LOGSTORE = "mcp-writeback-full";

/** SOC-searchable marker: rides the guarded request arguments only. */
const MARKER = "mcp-soc-probe-7f3a";

// Hard negatives: dot-separated numbers that are NOT version values.
const NEG = "Elapsed: 12.345s Memory: 4.2 GB 0.13um top.v:12:1 10.2.255.1";

// Request-side text (en + zh hits + negatives + the SOC marker).
const ARG_TEXT = `${MARKER} build version: 12.1 ${NEG} 工具版本：2022.4 完成`;
const ARG_MASKED = `${MARKER} build version: *** ${NEG} 工具版本：*** 完成`;

// Fixed `report` tool content (response side).
const SUMMARY = `阶段汇总 版本：2022.4 用时 12.345s`;
const LOG = `Compile OK version: 12.1\n${NEG}`;
const STRUCT_LOG = `config {"version": "12.1"} ok`;

interface RpcReply {
  status: number;
  body: string;
  json?: {
    result?: {
      content?: Array<{
        type: string;
        text?: string;
        resource?: { uri?: string; mimeType?: string; text?: string };
      }>;
      structuredContent?: Record<string, unknown>;
      isError?: boolean;
    };
    error?: { code: number; message: string };
  };
}

describe("mcp mask write-back e2e: /mcp", () => {
  let appG: SpawnedApp | undefined;
  let appP: SpawnedApp | undefined;
  let upstream: McpUpstream | undefined;
  let sls: MockSls | undefined;
  let etcdReachable = false;

  const post = async (app: SpawnedApp, body: unknown): Promise<RpcReply> => {
    const res = await fetch(`${app.proxyUrl}/mcp`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${KEY}`,
        "content-type": "application/json",
        accept: "application/json, text/event-stream",
      },
      body: JSON.stringify(body),
    });
    const text = await res.text();
    let json: RpcReply["json"];
    try {
      json = text ? (JSON.parse(text) as RpcReply["json"]) : undefined;
    } catch {
      json = undefined;
    }
    return { status: res.status, body: text, json };
  };

  /** Per-operation handshake; both apps serve the stateless endpoint. */
  const callTool = async (
    app: SpawnedApp,
    name: string,
    args: Record<string, unknown>,
  ): Promise<RpcReply> => {
    await post(app, {
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: {
        protocolVersion: "2025-11-25",
        capabilities: {},
        clientInfo: { name: "mcp-writeback-e2e", version: "0.1" },
      },
    });
    return post(app, {
      jsonrpc: "2.0",
      id: 3,
      method: "tools/call",
      params: { name, arguments: args },
    });
  };

  /** Upstream JSON-RPC ids are minted per ephemeral client; strip them so
   * the request byte-diff compares everything else exactly. */
  const stripIds = (s: string) => s.replace(/"id":\s*(?:"[^"]*"|\d+)/g, '"id":0');

  const seedEnv = async (
    app: SpawnedApp,
    opts: { guarded: boolean },
  ): Promise<void> => {
    const seed = new SeedClient(new EtcdClient(), app.etcdPrefix);
    await seed.update("mcp_servers", randomUUID(), {
      display_name: "eda",
      url: upstream!.url,
      enabled: true,
    });
    if (opts.guarded) {
      await seed.createObservabilityExporter({
        name: "sls-mcp-writeback",
        enabled: true,
        kind: "aliyun_sls",
        endpoint: sls!.url,
        project: SLS_PROJECT,
        logstore: FULL_LOGSTORE,
        credential_ref: CREDENTIAL_REF,
        content_mode: "full",
      });
      await seed.createGuardrail({
        name: "mcp-writeback-guard",
        enabled: true,
        hook_point: "both",
        kind: "pii",
        custom_patterns: [
          {
            name: "eda_version",
            regex: "(?:version|版本)\\s*[:：]\\s*(\\d+(?:\\.\\d+)+)",
            action: "mask",
            replacement: "***",
          },
          {
            name: "eda_version_json",
            regex: '"version"\\s*:\\s*"([^"]*)"',
            action: "mask",
            replacement: "***",
          },
        ],
      });
    }
    // Caller key LAST (AGENTS.md gate rule): the key authenticating
    // implies every row above is in the snapshot.
    await seed.createApiKey({
      key_hash: sha256(KEY),
      allowed_models: [],
      mcp_access: { allow: ["*"] },
    });
    const proxy = new ProxyClient(app.proxyUrl, KEY);
    await waitConfigPropagation(async () => (await proxy.listModels()).status === 200);
  };

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startMcpUpstream("eda", {
      reportContent: { summary: SUMMARY, log: LOG, structuredLog: STRUCT_LOG },
    });
    sls = await startMockSls();
    appG = await spawnApp({
      extraEnv: {
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: "mock-akid",
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: "mock-secret",
      },
    });
    appP = await spawnApp();
    await seedEnv(appG, { guarded: true });
    await seedEnv(appP, { guarded: false });
  }, 90_000);

  afterAll(async () => {
    await appG?.exit();
    await appP?.exit();
    await upstream?.close();
    await sls?.close();
  });

  test("request: upstream receives the masked arguments, byte-identical elsewhere", async (ctx) => {
    if (!etcdReachable || !appG || !appP || !upstream) return ctx.skip();

    const before = upstream.received.length;
    const guarded = await callTool(appG, "eda__echo", { text: ARG_TEXT });
    const baseline = await callTool(appP, "eda__echo", { text: ARG_TEXT });
    expect(guarded.status).toBe(200);
    expect(baseline.status).toBe(200);

    // The two upstream-received tools/call bodies differ ONLY in the
    // masked spans (and per-connection rpc ids, normalised out).
    const calls = upstream.received
      .slice(before)
      .filter((b) => b.includes('"tools/call"'));
    expect(calls).toHaveLength(2);
    const [guardedRaw, baselineRaw] = calls;
    expect(stripIds(guardedRaw)).toBe(
      stripIds(baselineRaw)
        .replace("version: 12.1", "version: ***")
        .replace("版本：2022.4", "版本：***"),
    );
    // The raw values never reached the upstream on the guarded path.
    expect(guardedRaw).not.toContain("version: 12.1");
    expect(guardedRaw).not.toContain("2022.4");

    // The echo reply reflects what the upstream actually saw: the masked
    // text — and the negatives byte-identical inside it.
    expect(guarded.json?.result?.isError).toBeFalsy();
    expect(guarded.json?.result?.content?.[0]?.text).toBe(`eda:${ARG_MASKED}`);
    expect(baseline.json?.result?.content?.[0]?.text).toBe(`eda:${ARG_TEXT}`);
  });

  test("response: text + embedded resource + structuredContent masked in place, full-body byte-diff, still JSON", async (ctx) => {
    if (!etcdReachable || !appG || !appP) return ctx.skip();

    const guarded = await callTool(appG, "eda__report", { text: "go" });
    const baseline = await callTool(appP, "eda__report", { text: "go" });
    expect(guarded.status).toBe(200);
    expect(baseline.status).toBe(200);

    // Rewrite never blocks: 200, no protocol error, no tool error.
    expect(guarded.json?.error).toBeUndefined();
    expect(guarded.json?.result?.isError).toBeFalsy();

    // Full-body byte-diff: both apps return the same client-facing bytes
    // (same request id, same upstream content) except the masked spans.
    // In the raw body the structuredContent hit appears JSON-escaped.
    expect(guarded.body).toBe(
      baseline.body
        .replace("version: 12.1", "version: ***") // resource.text log line
        .replace("版本：2022.4", "版本：***") // summary text block
        .replace('{\\"version\\": \\"12.1\\"}', '{\\"version\\": \\"***\\"}'), // structured leaf
    );
    expect(guarded.body).not.toContain("12.1");
    expect(guarded.body).not.toContain("2022.4");

    // Structural acceptance: parse the rewritten body, don't just diff it.
    const result = guarded.json?.result;
    expect(result?.content?.[0]?.text).toBe("阶段汇总 版本：*** 用时 12.345s");
    const resource = result?.content?.[1]?.resource;
    expect(resource?.mimeType).toBe("text/plain");
    expect(resource?.text).toBe(`Compile OK version: ***\n${NEG}`);
    expect(result?.structuredContent).toEqual({
      log: 'config {"version": "***"} ok',
      cells: 42,
    });
  });

  test("SOC export: captured content is the post-mask text with detector counts, never the raw values", async (ctx) => {
    if (!etcdReachable || !appG || !sls) return ctx.skip();

    // The guarded echo call from the request test carried the MARKER; its
    // usage event (with captured content) lands on the full logstore.
    await waitForToken(sls, FULL_LOGSTORE, MARKER);
    const decoded = decodedTextFor(sls, FULL_LOGSTORE);
    // Post-mask capture on both directions...
    expect(decoded).toContain("version: ***");
    expect(decoded).toContain("版本：***");
    // ...the detector name rides the event (counts, names only)...
    expect(decoded).toContain("eda_version");
    // ...and the raw values never reach the SOC target.
    expect(decoded).not.toContain("version: 12.1");
    expect(decoded).not.toContain("2022.4");
  });
});
