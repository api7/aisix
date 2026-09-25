import { createHash, randomUUID } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  spawnApp,
  startMcpUpstream,
  startMockSls,
  startOpenAiUpstream,
  waitConfigPropagation,
  waitForSlsLog,
  type McpUpstream,
  type MockSls,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: every model-serving request leaves a usage event — the record the
// console Logs and budgets read — even when there is nothing to bill:
//   - a Cohere rerank, whose upstream reports search units and no tokens;
//   - a request the gateway answers 501 itself because the provider lacks
//     the capability, so no upstream call is made;
//   - an MCP tool call whose upstream result the gateway cannot read back
//     (it outgrows the body cap), answered 502 after the call went out.
// Each records zero tokens.
//
// The body cap is lowered so the MCP tool result outgrows it.
const BODY_LIMIT = 65_536;

const CALLER = "sk-usage-event-every-request";
const CALLER_HASH = createHash("sha256").update(CALLER).digest("hex");
const CREDENTIAL_REF = "mock";
const LOGSTORE = "usage-event-every-request";
const auth = { authorization: `Bearer ${CALLER}`, "content-type": "application/json" };

// Cohere `rerank-v3.5`'s live response shape: no `input_tokens`.
const COHERE_RERANK = {
  id: "rerank-cohere-live",
  results: [
    { index: 1, relevance_score: 0.91 },
    { index: 0, relevance_score: 0.12 },
  ],
  meta: { api_version: { version: "2" }, billed_units: { search_units: 1 } },
};

describe("a usage event for every request", () => {
  let app: SpawnedApp | undefined;
  let sls: MockSls | undefined;
  let cohere: OpenAiUpstream | undefined;
  let mcp: McpUpstream | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    sls = await startMockSls();
    cohere = await startOpenAiUpstream({ nonStreamBody: COHERE_RERANK });
    mcp = await startMcpUpstream("big", {
      reportContent: { summary: "done", log: "x".repeat(4 * BODY_LIMIT), structuredLog: "ok" },
    });
    app = await spawnApp({
      requestBodyLimitBytes: BODY_LIMIT,
      extraEnv: {
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: "mock-akid",
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: "mock-secret",
      },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createObservabilityExporter({
      name: "sls-usage-event-every-request",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: "aisix-e2e-obs",
      logstore: LOGSTORE,
      credential_ref: CREDENTIAL_REF,
      content_mode: "metadata_only",
    });

    const coherePk = await seed.createProviderKey({
      display_name: "every-request-cohere",
      secret: "cohere-mock-key",
      provider: "cohere",
      adapter: "openai",
      api_base: cohere.baseUrl,
    });
    await seed.createModel({
      display_name: "every-request-rerank",
      provider: "cohere",
      model_name: "rerank-v3.5",
      provider_key_id: coherePk.id,
    });

    // Anthropic serves neither legacy completions nor embeddings; the
    // gateway answers those itself, without an upstream call.
    const anthropicPk = await seed.createProviderKey({
      display_name: "every-request-anthropic",
      secret: "sk-ant-mock",
      provider: "anthropic",
      api_base: "http://127.0.0.1:9",
    });
    await seed.createModel({
      display_name: "every-request-claude",
      provider: "anthropic",
      model_name: "claude-3-5-haiku-20241022",
      provider_key_id: anthropicPk.id,
    });

    // A guardrail on the MCP server makes the gateway read the tool result
    // back before relaying it.
    const mcpServerId = randomUUID();
    await seed.update("mcp_servers", mcpServerId, { display_name: "big", url: mcp.url, enabled: true });
    const monitor = await seed.createGuardrail(
      {
        name: "every-request-mcp-monitor",
        enabled: true,
        hook_point: "output",
        enforcement_mode: "monitor",
        kind: "keyword",
        patterns: [{ kind: "literal", value: "never-present-literal" }],
      },
      { attach: false },
    );
    await seed.update("guardrail_attachments", randomUUID(), {
      guardrail_id: monitor.id,
      scope_type: "mcp_server",
      scope_id: mcpServerId,
      priority: 100,
    });

    // Seeded last: its key authenticating implies the whole seed is live.
    await seed.createApiKey({ key_hash: CALLER_HASH, allowed_models: ["*"], mcp_access: { allow: ["*"] } });
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, { headers: auth });
      await res.arrayBuffer();
      return res.status === 200;
    });
  }, 60_000);

  afterAll(async () => {
    await app?.exit();
    await cohere?.close();
    await mcp?.close();
    await sls?.close();
  });

  const post = async (path: string, body: Record<string, unknown>) => {
    const res = await fetch(`${app!.proxyUrl}${path}`, { method: "POST", headers: auth, body: JSON.stringify(body) });
    return { status: res.status, body: await res.text() };
  };

  const eventFor = (model: string, operation: string) =>
    waitForSlsLog(
      sls!,
      LOGSTORE,
      (log) => log.get("requested_model") === model && log.get("operation") === operation,
      `${operation} usage event for ${model}`,
    );

  test("a Cohere rerank that reports only search units records a zero-token event", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const res = await post("/v1/rerank", { model: "every-request-rerank", query: "q", documents: ["a", "b"] });
    expect(res.status, res.body).toBe(200);
    expect(JSON.parse(res.body).results).toHaveLength(2);

    const event = await eventFor("every-request-rerank", "rerank");
    expect(event.get("status_code")).toBe("200");
    expect(event.get("prompt_tokens") ?? "0").toBe("0");
    expect(event.get("completion_tokens") ?? "0").toBe("0");
  });

  for (const [path, operation, body] of [
    ["/v1/completions", "completions", { model: "every-request-claude", prompt: "hi" }],
    ["/v1/embeddings", "embeddings", { model: "every-request-claude", input: "hi" }],
  ] as const) {
    test(`${path}: the gateway's own 501 records a zero-token event`, async (ctx) => {
      if (!etcdReachable || !app || !sls) return ctx.skip();
      const res = await post(path, body);
      expect(res.status, res.body).toBe(501);

      const event = await eventFor("every-request-claude", operation);
      expect(event.get("status_code")).toBe("501");
      expect(event.get("prompt_tokens") ?? "0").toBe("0");
    });
  }

  test("an MCP tool result the gateway cannot read back records a 502 event", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const rpc = (id: number, method: string, params: Record<string, unknown>) =>
      fetch(`${app!.proxyUrl}/mcp`, {
        method: "POST",
        headers: { ...auth, accept: "application/json, text/event-stream" },
        body: JSON.stringify({ jsonrpc: "2.0", id, method, params }),
      });
    await (
      await rpc(1, "initialize", {
        protocolVersion: "2025-11-25",
        capabilities: {},
        clientInfo: { name: "usage-event-every-request", version: "0.1" },
      })
    ).text();
    const res = await rpc(2, "tools/call", { name: "big__report", arguments: {} });
    const body = await res.text();
    expect(res.status, body).toBe(502);

    const event = await waitForSlsLog(
      sls,
      LOGSTORE,
      (log) => log.get("operation") === "mcp" && log.get("mcp_tool_name") === "report",
      "mcp usage event for big__report",
    );
    expect(event.get("status_code")).toBe("502");
    expect(event.get("mcp_server_name")).toBe("big");
  });
});
