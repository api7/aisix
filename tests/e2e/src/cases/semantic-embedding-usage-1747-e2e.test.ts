import { createHash, randomUUID } from "node:crypto";
import { createServer, type Server } from "node:http";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  slsLogsFor,
  spawnApp,
  startMockSls,
  startOpenAiUpstream,
  waitConfigPropagation,
  waitForSlsLog,
  type MockSls,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";
import { pickFreePort } from "../harness/ports.js";

// AISIX-Cloud#1747: semantic route selection, semantic cache lookups, and
// semantic guardrails all dispatch embeddings the caller did not ask for.
// They must remain CHILD work on the one real parent UsageEvent: count and
// token/latency visibility without changing the chat model's cost, token, or
// attempt attribution. This runs a real aisix DP + etcd and reads the
// metadata-only SLS export it emits; only the two upstream protocols are
// deterministic local endpoints.

const CALLER_PLAINTEXT = "sk-semantic-embedding-usage-1747";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const CREDENTIAL_REF = "mock";
const SLS_PROJECT = "aisix-e2e-obs";
const LOGSTORE = "semantic-embedding-usage-1747";
const EMBEDDING_DELAY_MS = 20;

const EMBED_MODEL = "seu-1747-embed";
const ROUTER_MODEL = "seu-1747-router";
const ROUTE_TARGET = "seu-1747-route-target";
const ROUTE_DEFAULT = "seu-1747-route-default";
const GUARDRAIL_MODEL = "seu-1747-guardrail-chat";
const CACHE_MODEL = "seu-1747-cache-chat";

const ROUTE_EXAMPLE = "route-topic prototype";
const ROUTE_PROMPT = "route-topic caller question";
const GUARDRAIL_EXAMPLES = [
  "guardrail-prototype-first",
  "guardrail-prototype-second",
];
const GUARDRAIL_PROMPT = "guardrail-candidate-allowed";
const CACHE_PROMPT = "cache-topic exact-hit";

function keywordVector(text: string): number[] {
  const lower = text.toLowerCase();
  if (lower.includes("route-topic")) return [1, 0, 0, 0];
  if (lower.includes("guardrail-prototype")) return [0, 1, 0, 0];
  if (lower.includes("guardrail-candidate")) return [0, 0, 1, 0];
  if (lower.includes("cache-topic")) return [0, 0, 0, 1];
  return [0.5, 0.5, 0.5, 0.5];
}

interface EmbeddingMock {
  baseUrl: string;
  callCount(): number;
  close(): Promise<void>;
}

/** OpenAI-compatible embedding endpoint with provider-reported usage. */
async function startEmbeddingMock(): Promise<EmbeddingMock> {
  let calls = 0;
  const server: Server = createServer((req, res) => {
    res.on("error", () => {});
    let raw = "";
    req.on("data", (chunk: Buffer) => (raw += chunk.toString("utf8")));
    req.on("end", () => {
      if (!req.url?.includes("/embeddings")) {
        res.statusCode = 404;
        res.end("{}");
        return;
      }

      let body: { input?: string | string[] };
      try {
        body = JSON.parse(raw || "{}") as { input?: string | string[] };
      } catch {
        res.statusCode = 400;
        res.setHeader("content-type", "application/json");
        res.end(JSON.stringify({ error: { message: "invalid JSON" } }));
        return;
      }

      calls++;
      const inputs = Array.isArray(body.input) ? body.input : [body.input ?? ""];
      const promptTokens = inputs.length * 10;
      setTimeout(() => {
        res.statusCode = 200;
        res.setHeader("content-type", "application/json");
        res.end(
          JSON.stringify({
            object: "list",
            model: "embedding-usage-mock",
            data: inputs.map((text, index) => ({
              object: "embedding",
              index,
              embedding: keywordVector(text),
            })),
            usage: {
              prompt_tokens: promptTokens,
              total_tokens: promptTokens + 1,
            },
          }),
        );
      }, EMBEDDING_DELAY_MS);
    });
  });
  const port = await pickFreePort();
  await new Promise<void>((resolve) => server.listen(port, "127.0.0.1", resolve));
  return {
    baseUrl: `http://127.0.0.1:${port}`,
    callCount: () => calls,
    async close() {
      await new Promise<void>((resolve, reject) => {
        server.close((err) => (err ? reject(err) : resolve()));
      });
    },
  };
}

interface ChatResponse {
  status: number;
  requestId: string;
  cache: string | null;
  cacheLayer: string | null;
  route: string | null;
  servedBy: string | null;
}

interface GatewayEmbeddingCall {
  count: number;
  purpose: "semantic_route" | "semantic_cache" | "guardrail";
  embedding_model_id: string;
  prompt_tokens: number;
  total_tokens: number;
  latency_ms: number;
  outcome: "succeeded" | "failed";
}

const rowsForRequest = (sls: MockSls, requestId: string) =>
  slsLogsFor(sls, LOGSTORE).filter((row) => row.get("request_id") === requestId);

function embeddingCalls(row: Map<string, string>): GatewayEmbeddingCall[] {
  const raw = row.get("gateway_embedding_calls");
  expect(raw, `usage row carried no embedding audit: ${[...row.keys()].join(",")}`).toBeDefined();
  return JSON.parse(raw!) as GatewayEmbeddingCall[];
}

function expectSucceededCall(
  call: GatewayEmbeddingCall,
  purpose: GatewayEmbeddingCall["purpose"],
  embeddingModelID: string,
  promptTokens: number,
): void {
  expect(call.count).toBe(1);
  expect(call.purpose).toBe(purpose);
  expect(call.embedding_model_id).toBe(embeddingModelID);
  // These are exactly the response's provider-reported values, not a local
  // estimate: the mock returns 10 tokens per input and total = prompt + 1.
  expect(call.prompt_tokens).toBe(promptTokens);
  expect(call.total_tokens).toBe(promptTokens + 1);
  // The endpoint delays every response, making this assert a real duration
  // rather than merely a field's existence.
  expect(call.latency_ms).toBeGreaterThan(0);
  expect(call.outcome).toBe("succeeded");
}

function expectUnchangedParentUsage(row: Map<string, string>): void {
  // The chat upstream's own usage remains the parent usage. The embedding
  // model must not inflate its token totals or the cost field in Phase 1.
  expect(row.get("prompt_tokens")).toBe("11");
  expect(row.get("completion_tokens")).toBe("7");
  expect(row.get("total_tokens")).toBe("18");
  expect(Number(row.get("cost_usd"))).toBe(0);
}

describe("gateway-initiated embedding usage on the real parent event (#1747)", () => {
  let app: SpawnedApp | undefined;
  let sls: MockSls | undefined;
  let embed: EmbeddingMock | undefined;
  let upstream: OpenAiUpstream | undefined;
  let etcdReachable = false;
  let embeddingModelID = "";
  let routeTargetModelID = "";
  let guardrailModelID = "";
  let cacheModelID = "";

  async function chat(model: string, prompt: string): Promise<ChatResponse> {
    const res = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
      },
      body: JSON.stringify({
        model,
        messages: [{ role: "user", content: prompt }],
      }),
    });
    await res.text();
    return {
      status: res.status,
      requestId: res.headers.get("x-aisix-request-id") ?? "",
      cache: res.headers.get("x-aisix-cache"),
      cacheLayer: res.headers.get("x-aisix-cache-layer"),
      route: res.headers.get("x-aisix-route"),
      servedBy: res.headers.get("x-aisix-served-by"),
    };
  }

  async function usageRow(requestId: string): Promise<Map<string, string>> {
    expect(requestId, "the DP stamps the parent request id").not.toBe("");
    return waitForSlsLog(
      sls!,
      LOGSTORE,
      (row) => row.get("request_id") === requestId,
      `the parent UsageEvent for ${requestId}`,
    );
  }

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    sls = await startMockSls();
    embed = await startEmbeddingMock();
    upstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "cmpl-semantic-embedding-usage",
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: "gpt-4o-mini",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "upstream-answered" },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 11, completion_tokens: 7, total_tokens: 18 },
      },
    });
    app = await spawnApp({
      extraEnv: {
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: "mock-akid",
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: "mock-secret",
      },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);

    await seed.createObservabilityExporter({
      name: "semantic-embedding-usage-sls",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: SLS_PROJECT,
      logstore: LOGSTORE,
      credential_ref: CREDENTIAL_REF,
      content_mode: "metadata_only",
    });

    const embeddingKey = await seed.createProviderKey({
      display_name: "semantic-embedding-usage-embed-pk",
      secret: "sk-embedding-mock",
      api_base: `${embed.baseUrl}/v1`,
    });
    const embedding = await seed.createModel({
      display_name: EMBED_MODEL,
      provider: "openai",
      model_name: "embedding-usage-mock",
      provider_key_id: embeddingKey.id,
      embedding: { dimensions: 4, normalize: true },
    });
    embeddingModelID = embedding.id;

    const chatKey = await seed.createProviderKey({
      display_name: "semantic-embedding-usage-chat-pk",
      secret: "sk-chat-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    const createChatModel = (displayName: string) =>
      seed.createModel({
        display_name: displayName,
        provider: "openai",
        model_name: "gpt-4o-mini",
        provider_key_id: chatKey.id,
      });

    const routeTarget = await createChatModel(ROUTE_TARGET);
    routeTargetModelID = routeTarget.id;
    await createChatModel(ROUTE_DEFAULT);
    const guardrailModel = await createChatModel(GUARDRAIL_MODEL);
    guardrailModelID = guardrailModel.id;
    const cacheModel = await createChatModel(CACHE_MODEL);
    cacheModelID = cacheModel.id;

    await seed.createModel({
      display_name: ROUTER_MODEL,
      semantic: {
        embedding_model: EMBED_MODEL,
        routes: [
          {
            name: "route-topic",
            target: ROUTE_TARGET,
            examples: [ROUTE_EXAMPLE],
            threshold: 0.9,
          },
        ],
        default: ROUTE_DEFAULT,
        match: { threshold: 0.9 },
      },
    });

    const semanticGuardrail = await seed.createGuardrail(
      {
        enabled: true,
        name: "semantic-embedding-usage-guardrail",
        hook_point: "input",
        enforcement_mode: "block",
        kind: "semantic",
        embedding_model: EMBED_MODEL,
        deny_examples: GUARDRAIL_EXAMPLES,
        deny_threshold: 0.99,
      },
      { attach: false },
    );
    await seed.update("guardrail_attachments", randomUUID(), {
      guardrail_id: semanticGuardrail.id,
      scope_type: "model",
      scope_id: guardrailModelID,
      priority: 100,
    });

    await seed.createCachePolicy({
      name: "semantic-embedding-usage-cache",
      backend: "memory",
      applies_to: `model:${CACHE_MODEL}`,
      ttl_seconds: 600,
      semantic: { embedding_model: EMBED_MODEL, threshold: 0.9 },
    });

    // The API key lands last; authenticating it proves this complete etcd
    // revision has propagated without using behavior under test as readiness.
    await seed.createApiKey({ key_hash: CALLER_KEY_HASH, allowed_models: ["*"] });
    const probe = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    await waitConfigPropagation(async () => (await probe.listModels()).status === 200);
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    await embed?.close();
    await sls?.close();
  });

  test("semantic routing records its real batched embedding on the chat parent", async (ctx) => {
    if (!etcdReachable || !app || !sls || !embed) {
      ctx.skip();
      return;
    }

    const callsBefore = embed.callCount();
    const response = await chat(ROUTER_MODEL, ROUTE_PROMPT);
    expect(response.status).toBe(200);
    expect(response.route).toBe("route-topic");
    expect(response.servedBy).toBe(ROUTE_TARGET);
    expect(embed.callCount()).toBe(callsBefore + 1);

    const row = await usageRow(response.requestId);
    expect(rowsForRequest(sls, response.requestId)).toHaveLength(1);
    const calls = embeddingCalls(row);
    expect(calls).toHaveLength(1);
    expectSucceededCall(calls[0]!, "semantic_route", embeddingModelID, 20);
    expectUnchangedParentUsage(row);
    // The route target, not the detached embedding model, still owns the
    // parent event's pricing and attempt attribution.
    expect(row.get("model_id")).toBe(routeTargetModelID);
    expect(row.get("attempt_model")).toBe(ROUTE_TARGET);
    expect(row.get("attempt_model")).not.toBe(EMBED_MODEL);
  });

  test("semantic guardrail records prototypes and candidate only on its parent", async (ctx) => {
    if (!etcdReachable || !app || !sls || !embed) {
      ctx.skip();
      return;
    }

    const callsBefore = embed.callCount();
    const response = await chat(GUARDRAIL_MODEL, GUARDRAIL_PROMPT);
    expect(response.status).toBe(200);
    // A cold semantic guardrail makes one cacheable prototype batch and one
    // request-text batch, so two actual bridge calls must be visible.
    expect(embed.callCount()).toBe(callsBefore + 2);

    const row = await usageRow(response.requestId);
    expect(rowsForRequest(sls, response.requestId)).toHaveLength(1);
    const calls = embeddingCalls(row);
    expect(calls).toHaveLength(2);
    const byPromptTokens = [...calls].sort((a, b) => a.prompt_tokens - b.prompt_tokens);
    expectSucceededCall(byPromptTokens[0]!, "guardrail", embeddingModelID, 10);
    expectSucceededCall(byPromptTokens[1]!, "guardrail", embeddingModelID, 20);
    expectUnchangedParentUsage(row);
    expect(row.get("model_id")).toBe(guardrailModelID);

    // `guardrail-semantic` remains bridge-context only. It must never be
    // emitted as a fabricated child request, and neither screened text nor
    // policy examples belong in the durable child audit field.
    expect(rowsForRequest(sls, "guardrail-semantic")).toHaveLength(0);
    const childAudit = row.get("gateway_embedding_calls")!;
    for (const text of [...GUARDRAIL_EXAMPLES, GUARDRAIL_PROMPT]) {
      expect(childAudit).not.toContain(text);
    }
  });

  test("semantic-cache exact hit makes no embedding call and carries no child audit", async (ctx) => {
    if (!etcdReachable || !app || !sls || !embed) {
      ctx.skip();
      return;
    }

    const callsBeforeMiss = embed.callCount();
    const miss = await chat(CACHE_MODEL, CACHE_PROMPT);
    expect(miss.status).toBe(200);
    expect(miss.cache).toBe("miss");
    expect(embed.callCount()).toBe(callsBeforeMiss + 1);

    const missRow = await usageRow(miss.requestId);
    expect(rowsForRequest(sls, miss.requestId)).toHaveLength(1);
    const missCalls = embeddingCalls(missRow);
    expect(missCalls).toHaveLength(1);
    expectSucceededCall(missCalls[0]!, "semantic_cache", embeddingModelID, 10);
    expectUnchangedParentUsage(missRow);
    expect(missRow.get("model_id")).toBe(cacheModelID);

    const callsBeforeHit = embed.callCount();
    const hit = await chat(CACHE_MODEL, CACHE_PROMPT);
    expect(hit.status).toBe(200);
    expect(hit.cache).toBe("hit");
    expect(hit.cacheLayer).toBe("exact");
    expect(embed.callCount()).toBe(callsBeforeHit);

    const hitRow = await usageRow(hit.requestId);
    expect(rowsForRequest(sls, hit.requestId)).toHaveLength(1);
    expect(hitRow.get("gateway_embedding_calls")).toBeUndefined();
  });
});
