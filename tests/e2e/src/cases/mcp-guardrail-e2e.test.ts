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

// E2E: guardrails over the MCP gateway, against a real gateway + etcd + two
// real MCP upstreams (official TypeScript SDK servers).
//
// Pinned contract:
//   - a policy rejection is a TOOL-EXECUTION error (`result.isError`), never a
//     JSON-RPC protocol error, so the calling agent reads it as tool output;
//   - `structuredContent` is scanned, not just the text content blocks — the
//     gateway relays that field to the client verbatim, so a tool that returns
//     clean prose plus a sensitive structured payload must not slip through;
//   - an `mcp_server`-scoped attachment governs only the tool calls routed to
//     that server, the dimension no model scope can express for MCP.

const KEY = "sk-mcp-guardrail-e2e";
const sha256 = (s: string) => createHash("sha256").update(s).digest("hex");

/** Distinct patterns so the three guardrails can coexist in one env. */
const ALPHA_ONLY_PATTERN = "poison-alpha-only";
const EVERYWHERE_PATTERN = "blocked-everywhere";
const STRUCTURED_PATTERN = "classified-payload";

interface RpcReply {
  status: number;
  json?: {
    result?: {
      content?: Array<{ type: string; text?: string }>;
      structuredContent?: Record<string, unknown>;
      isError?: boolean;
    };
    error?: { code: number; message: string };
  };
}

describe("mcp guardrails e2e: /mcp", () => {
  let app: SpawnedApp | undefined;
  let alpha: McpUpstream | undefined;
  let beta: McpUpstream | undefined;
  let etcdReachable = false;
  let seed: SeedClient;

  const post = async (body: unknown): Promise<RpcReply> => {
    const res = await fetch(`${app!.proxyUrl}/mcp`, {
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
    return { status: res.status, json };
  };

  /** Spec-faithful per-operation handshake (the endpoint is stateless). */
  const callTool = async (name: string, text: string): Promise<RpcReply> => {
    await post({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: {
        protocolVersion: "2025-11-25",
        capabilities: {},
        clientInfo: { name: "mcp-guardrail-e2e", version: "0.1" },
      },
    });
    return post({
      jsonrpc: "2.0",
      id: 3,
      method: "tools/call",
      params: { name, arguments: { text } },
    });
  };

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    alpha = await startMcpUpstream("alpha", { structuredTool: true });
    beta = await startMcpUpstream("beta");
    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);

    const alphaId = randomUUID();
    await seed.update("mcp_servers", alphaId, {
      display_name: "alpha",
      url: alpha.url,
      enabled: true,
    });
    await seed.update("mcp_servers", randomUUID(), {
      display_name: "beta",
      url: beta.url,
      enabled: true,
    });
    // One guardrail per behaviour under test, each with its own pattern, so
    // all three can be attached at once without interfering.
    const alphaOnly = await seed.createGuardrail({
      name: "mcp-alpha-only",
      kind: "keyword",
      patterns: [{ kind: "literal", value: ALPHA_ONLY_PATTERN }],
    });
    const everywhere = await seed.createGuardrail({
      name: "mcp-everywhere",
      kind: "keyword",
      patterns: [{ kind: "literal", value: EVERYWHERE_PATTERN }],
    });
    const structured = await seed.createGuardrail({
      name: "mcp-structured",
      kind: "keyword",
      hook_point: "output",
      patterns: [{ kind: "literal", value: STRUCTURED_PATTERN }],
    });

    await seed.update("guardrail_attachments", randomUUID(), {
      guardrail_id: alphaOnly.id,
      scope_type: "mcp_server",
      scope_id: alphaId,
      priority: 100,
    });
    for (const id of [everywhere.id, structured.id]) {
      await seed.update("guardrail_attachments", randomUUID(), {
        guardrail_id: id,
        scope_type: "env",
        priority: 50,
      });
    }

    // The caller key is written LAST, so the key authenticating implies every
    // row above it is already in the snapshot (etcd applies in revision
    // order). The gate deliberately touches neither `/mcp` nor a guardrail:
    // a broken assertion must fail as an assertion, not as a gate timeout.
    await seed.createApiKey({
      key_hash: sha256(KEY),
      allowed_models: [],
      mcp_access: { allow: ["*"] },
    });
    const proxy = new ProxyClient(app.proxyUrl, KEY);
    await waitConfigPropagation(async () => (await proxy.listModels()).status === 200);
  }, 60_000);

  afterAll(async () => {
    await app?.exit();
    await alpha?.close();
    await beta?.close();
  });

  test("a policy rejection is a tool error, not a protocol error", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    const reply = await callTool("beta__echo", `say ${EVERYWHERE_PATTERN}`);

    expect(reply.status).toBe(200);
    expect(reply.json?.error).toBeUndefined();
    expect(reply.json?.result?.isError).toBe(true);
    expect(reply.json?.result?.content?.[0]?.text).toContain("content policy");
    // The firing rule is named so an operator can find it; the matched
    // content never is.
    expect(reply.json?.result?.content?.[0]?.text).toContain("mcp-everywhere");
    expect(reply.json?.result?.content?.[0]?.text).not.toContain(
      EVERYWHERE_PATTERN,
    );
  });

  test("structuredContent reaches the client, so it is scanned", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    // First: the field really is relayed verbatim. This is the reason it must
    // be scanned — without this leg, a passing block test could just mean the
    // gateway had dropped the field.
    const clean = await callTool("alpha__lookup", "ordinary value");
    expect(clean.json?.result?.isError).toBeFalsy();
    expect(clean.json?.result?.structuredContent).toEqual({
      record: { note: "ordinary value" },
    });

    // The text block `lookup` returns is constant and clean, so only the
    // structured payload carries the pattern.
    const blocked = await callTool("alpha__lookup", STRUCTURED_PATTERN);
    expect(blocked.json?.result?.isError).toBe(true);
    expect(blocked.json?.result?.content?.[0]?.text).toContain("tool result");
    expect(blocked.json?.result?.structuredContent).toBeUndefined();
  });

  test("an mcp_server scope governs only its own server", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    const onAlpha = await callTool("alpha__echo", `carrying ${ALPHA_ONLY_PATTERN}`);
    expect(onAlpha.json?.result?.isError).toBe(true);
    expect(onAlpha.json?.result?.content?.[0]?.text).toContain("mcp-alpha-only");

    // Same content, different server: the attachment does not reach it, and
    // the call runs through to the upstream.
    const onBeta = await callTool("beta__echo", `carrying ${ALPHA_ONLY_PATTERN}`);
    expect(onBeta.json?.result?.isError).toBeFalsy();
    expect(onBeta.json?.result?.content?.[0]?.text).toBe(
      `beta:carrying ${ALPHA_ONLY_PATTERN}`,
    );
  });
});
