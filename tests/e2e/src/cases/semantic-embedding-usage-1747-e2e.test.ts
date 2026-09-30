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
// The cancellation case must leave the bridge pending long enough for the
// client to disconnect after the real DP has dispatched it.
const CANCELLED_EMBEDDING_DELAY_MS = 30_000;

const EMBED_MODEL = "seu-1747-embed";
const ROUTER_MODEL = "seu-1747-router";
const ROUTE_TARGET = "seu-1747-route-target";
const ROUTE_DEFAULT = "seu-1747-route-default";
const STREAM_ROUTER_MODEL = "seu-1747-router-stream";
const STREAM_ROUTE_TARGET = "seu-1747-route-target-stream";
const GUARDRAIL_MODEL = "seu-1747-guardrail-chat";
const GUARDRAIL_OVERFLOW_MODEL = "seu-1747-guardrail-overflow-chat";
const CACHE_MODEL = "seu-1747-cache-chat";
const UNAVAILABLE_USAGE_EMBED_MODEL = "seu-1747-embed-usage-unavailable";
const UNAVAILABLE_USAGE_ROUTER_MODEL = "seu-1747-router-usage-unavailable";
const UNAVAILABLE_USAGE_UPSTREAM_MODEL = "embedding-usage-unavailable-mock";
const FAILED_USAGE_EMBED_MODEL = "seu-1747-embed-usage-failed";
const FAILED_USAGE_ROUTER_MODEL = "seu-1747-router-usage-failed";
const FALLBACK_USAGE_ROUTER_MODEL = "seu-1747-router-usage-fallback";
const FAILED_USAGE_UPSTREAM_MODEL = "embedding-usage-failed-mock";
const CANCELLED_EMBED_MODEL = "seu-1747-embed-cancelled";
const CANCELLED_ROUTER_MODEL = "seu-1747-router-cancelled";
const CANCELLED_USAGE_UPSTREAM_MODEL = "embedding-usage-cancelled-mock";
const OUTPUT_GUARDRAIL_ROUTER_MODEL = "seu-1747-output-guardrail-router";
const OUTPUT_GUARDRAIL_TARGET = "seu-1747-output-guardrail-target";
const GUARDRAIL_OVERFLOW_COUNT = 33;

const ROUTE_EXAMPLE = "route-topic prototype";
const OUTPUT_ROUTE_EXAMPLE = "output-route-topic prototype";
// Keep the streaming router's prototype distinct from the buffered router's.
// Semantic prototype embeddings are cached across requests, and sharing this
// value would turn the second scenario into a one-input call depending on test
// order rather than proving its own two-input bridge usage.
const STREAM_ROUTE_EXAMPLE = "stream-route-topic prototype";
const ROUTE_PROMPT = "route-topic caller question";
const OUTPUT_ROUTE_PROMPT = "output-route-topic caller question";
const STREAM_RESPONSE_TEXT = "streamed semantic route answer";
const STREAM_USAGE = {
  prompt_tokens: 13,
  completion_tokens: 5,
  total_tokens: 18,
};
const STREAM_EVENTS = [
  JSON.stringify({
    id: "cmpl-semantic-embedding-stream",
    object: "chat.completion.chunk",
    created: 1_700_000_000,
    model: "gpt-4o-mini",
    choices: [{ index: 0, delta: { role: "assistant" }, finish_reason: null }],
  }),
  JSON.stringify({
    id: "cmpl-semantic-embedding-stream",
    object: "chat.completion.chunk",
    created: 1_700_000_000,
    model: "gpt-4o-mini",
    choices: [{ index: 0, delta: { content: STREAM_RESPONSE_TEXT }, finish_reason: null }],
  }),
  JSON.stringify({
    id: "cmpl-semantic-embedding-stream",
    object: "chat.completion.chunk",
    created: 1_700_000_000,
    model: "gpt-4o-mini",
    choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
  }),
  JSON.stringify({
    id: "cmpl-semantic-embedding-stream",
    object: "chat.completion.chunk",
    created: 1_700_000_000,
    model: "gpt-4o-mini",
    choices: [],
    usage: STREAM_USAGE,
  }),
  "[DONE]",
];
const GUARDRAIL_EXAMPLES = [
  "guardrail-prototype-first",
  "guardrail-prototype-second",
];
const OUTPUT_GUARDRAIL_EXAMPLES = ["output-guardrail-prototype"];
const GUARDRAIL_PROMPT = "guardrail-candidate-allowed";
const GUARDRAIL_OVERFLOW_PROMPT = "guardrail-prototype-overflow-candidate";
const GUARDRAIL_OVERFLOW_PROTOTYPES = Array.from(
  { length: GUARDRAIL_OVERFLOW_COUNT },
  (_, index) => `guardrail-prototype-overflow-${index}`,
);
const CACHE_PROMPT = "cache-topic exact-hit";
const SLS_BARRIER_PROMPT = "semantic-embedding-usage-sls-barrier";

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
  callCountFor(model: string): number;
  close(): Promise<void>;
}

/** OpenAI-compatible embedding endpoint with provider-reported usage. */
async function startEmbeddingMock(): Promise<EmbeddingMock> {
  let calls = 0;
  const callsByModel = new Map<string, number>();
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

      let body: { model?: string; input?: string | string[] };
      try {
        body = JSON.parse(raw || "{}") as { model?: string; input?: string | string[] };
      } catch {
        res.statusCode = 400;
        res.setHeader("content-type", "application/json");
        res.end(JSON.stringify({ error: { message: "invalid JSON" } }));
        return;
      }

      calls++;
      const upstreamModel = body.model ?? "embedding-usage-mock";
      callsByModel.set(upstreamModel, (callsByModel.get(upstreamModel) ?? 0) + 1);
      if (body.model === FAILED_USAGE_UPSTREAM_MODEL) {
        res.statusCode = 502;
        res.setHeader("content-type", "application/json");
        res.end(JSON.stringify({ error: { message: "embedding upstream unavailable" } }));
        return;
      }
      const inputs = Array.isArray(body.input) ? body.input : [body.input ?? ""];
      const promptTokens = inputs.length * 10;
      const responseTimer = setTimeout(
        () => {
          if (res.destroyed || res.writableEnded) return;
          res.statusCode = 200;
          res.setHeader("content-type", "application/json");
          res.end(
            JSON.stringify({
              object: "list",
              model: upstreamModel,
              data: inputs.map((text, index) => ({
                object: "embedding",
                index,
                embedding: keywordVector(text),
              })),
              ...(body.model === UNAVAILABLE_USAGE_UPSTREAM_MODEL
                ? {}
                : {
                    usage: {
                      prompt_tokens: promptTokens,
                      total_tokens: promptTokens + 1,
                    },
                  }),
            }),
          );
        },
        body.model === CANCELLED_USAGE_UPSTREAM_MODEL
          ? CANCELLED_EMBEDDING_DELAY_MS
          : EMBEDDING_DELAY_MS,
      );
      // A cancelled DP bridge closes this response before the intentionally
      // slow timer fires. Clearing it keeps a failed test from pinning its
      // Node fixture for the full delay during teardown.
      res.once("close", () => clearTimeout(responseTimer));
    });
  });
  const port = await pickFreePort();
  await new Promise<void>((resolve) => server.listen(port, "127.0.0.1", resolve));
  return {
    baseUrl: `http://127.0.0.1:${port}`,
    callCount: () => calls,
    callCountFor: (model) => callsByModel.get(model) ?? 0,
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
  usage_source: "reported" | "unavailable";
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
  expect(call.usage_source).toBe("reported");
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
  let streamUpstream: OpenAiUpstream | undefined;
  let embeddingModelID = "";
  let unavailableUsageEmbeddingModelID = "";
  let failedUsageEmbeddingModelID = "";
  let failedUsageRouterModelID = "";
  let cancelledEmbeddingModelID = "";
  let routeTargetModelID = "";
  let routeDefaultModelID = "";
  let streamRouteTargetModelID = "";
  let outputGuardrailTargetModelID = "";
  let guardrailModelID = "";
  let overflowGuardrailModelID = "";
  let cacheModelID = "";

  function requireRuntime(): {
    app: SpawnedApp;
    sls: MockSls;
    embed: EmbeddingMock;
    upstream: OpenAiUpstream;
    streamUpstream: OpenAiUpstream;
  } {
    if (!app || !sls || !embed || !upstream || !streamUpstream) {
      throw new Error("semantic embedding usage e2e setup did not complete");
    }
    return { app, sls, embed, upstream, streamUpstream };
  }

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
    const row = await waitForSlsLog(
      sls!,
      LOGSTORE,
      (row) => row.get("request_id") === requestId,
      `the parent UsageEvent for ${requestId}`,
    );

    // The SLS writer is asynchronous. Its FIFO queue guarantees the direct
    // model's later row follows every row for this request, making exact-one
    // checks below stable rather than a snapshot of an in-flight export.
    const barrier = await chat(ROUTE_DEFAULT, SLS_BARRIER_PROMPT);
    expect(barrier.status, "the direct-model SLS barrier must complete").toBe(200);
    expect(barrier.requestId, "the direct-model SLS barrier must have a request id").not.toBe("");
    await waitForSlsLog(
      sls!,
      LOGSTORE,
      (barrierRow) => barrierRow.get("request_id") === barrier.requestId,
      `the SLS barrier UsageEvent for ${barrier.requestId}`,
    );
    return row;
  }

  beforeAll(async () => {
    const etcd = new EtcdClient();
    expect(await etcd.ping(), "semantic embedding usage e2e requires etcd").toBe(true);

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
    streamUpstream = await startOpenAiUpstream({
      eventDelayMs: 2,
      streamEvents: STREAM_EVENTS,
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
    const unavailableUsageEmbedding = await seed.createModel({
      display_name: UNAVAILABLE_USAGE_EMBED_MODEL,
      provider: "openai",
      model_name: UNAVAILABLE_USAGE_UPSTREAM_MODEL,
      provider_key_id: embeddingKey.id,
      embedding: { dimensions: 4, normalize: true },
    });
    unavailableUsageEmbeddingModelID = unavailableUsageEmbedding.id;
    const failedUsageEmbedding = await seed.createModel({
      display_name: FAILED_USAGE_EMBED_MODEL,
      provider: "openai",
      model_name: FAILED_USAGE_UPSTREAM_MODEL,
      provider_key_id: embeddingKey.id,
      embedding: { dimensions: 4, normalize: true },
    });
    failedUsageEmbeddingModelID = failedUsageEmbedding.id;
    const cancelledEmbedding = await seed.createModel({
      display_name: CANCELLED_EMBED_MODEL,
      provider: "openai",
      model_name: CANCELLED_USAGE_UPSTREAM_MODEL,
      provider_key_id: embeddingKey.id,
      embedding: { dimensions: 4, normalize: true },
    });
    cancelledEmbeddingModelID = cancelledEmbedding.id;

    const chatKey = await seed.createProviderKey({
      display_name: "semantic-embedding-usage-chat-pk",
      secret: "sk-chat-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    const streamChatKey = await seed.createProviderKey({
      display_name: "semantic-embedding-usage-stream-chat-pk",
      secret: "sk-stream-chat-mock",
      api_base: `${streamUpstream.baseUrl}/v1`,
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
    const streamRouteTarget = await seed.createModel({
      display_name: STREAM_ROUTE_TARGET,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: streamChatKey.id,
    });
    streamRouteTargetModelID = streamRouteTarget.id;
    const routeDefault = await createChatModel(ROUTE_DEFAULT);
    routeDefaultModelID = routeDefault.id;
    const guardrailModel = await createChatModel(GUARDRAIL_MODEL);
    guardrailModelID = guardrailModel.id;
    const overflowGuardrailModel = await createChatModel(GUARDRAIL_OVERFLOW_MODEL);
    overflowGuardrailModelID = overflowGuardrailModel.id;
    const cacheModel = await createChatModel(CACHE_MODEL);
    cacheModelID = cacheModel.id;
    const outputGuardrailTarget = await createChatModel(OUTPUT_GUARDRAIL_TARGET);
    outputGuardrailTargetModelID = outputGuardrailTarget.id;

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
    const failedUsageRouter = await seed.createModel({
      display_name: FAILED_USAGE_ROUTER_MODEL,
      semantic: {
        embedding_model: FAILED_USAGE_EMBED_MODEL,
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
        on_embedding_failure: "fail",
      },
    });
    failedUsageRouterModelID = failedUsageRouter.id;
    await seed.createModel({
      display_name: FALLBACK_USAGE_ROUTER_MODEL,
      semantic: {
        embedding_model: FAILED_USAGE_EMBED_MODEL,
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
    await seed.createModel({
      display_name: OUTPUT_GUARDRAIL_ROUTER_MODEL,
      semantic: {
        embedding_model: EMBED_MODEL,
        routes: [
          {
            name: "output-route-topic",
            target: OUTPUT_GUARDRAIL_TARGET,
            examples: [OUTPUT_ROUTE_EXAMPLE],
            threshold: 0.9,
          },
        ],
        default: ROUTE_DEFAULT,
        match: { threshold: 0.9 },
      },
    });
    await seed.createModel({
      display_name: STREAM_ROUTER_MODEL,
      semantic: {
        embedding_model: EMBED_MODEL,
        routes: [
          {
            name: "route-topic",
            target: STREAM_ROUTE_TARGET,
            examples: [STREAM_ROUTE_EXAMPLE],
            threshold: 0.9,
          },
        ],
        default: ROUTE_DEFAULT,
        match: { threshold: 0.9 },
      },
    });
    await seed.createModel({
      display_name: UNAVAILABLE_USAGE_ROUTER_MODEL,
      semantic: {
        embedding_model: UNAVAILABLE_USAGE_EMBED_MODEL,
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
    await seed.createModel({
      display_name: CANCELLED_ROUTER_MODEL,
      semantic: {
        embedding_model: CANCELLED_EMBED_MODEL,
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

    const outputSemanticGuardrail = await seed.createGuardrail(
      {
        enabled: true,
        name: "semantic-embedding-usage-output-guardrail",
        hook_point: "output",
        enforcement_mode: "monitor",
        kind: "semantic",
        embedding_model: EMBED_MODEL,
        deny_examples: OUTPUT_GUARDRAIL_EXAMPLES,
        deny_threshold: 0.99,
      },
      { attach: false },
    );
    await seed.attachGuardrailToModel(
      outputSemanticGuardrail.id,
      outputGuardrailTargetModelID,
      100,
    );

    // Each distinct prototype is cold, while the candidate is intentionally
    // uncached request data. Monitor mode lets every row run even though this
    // mock maps the prototype and candidate to the same vector.
    for (const [index, prototype] of GUARDRAIL_OVERFLOW_PROTOTYPES.entries()) {
      const guardrail = await seed.createGuardrail(
        {
          enabled: true,
          name: `semantic-embedding-usage-overflow-${index}`,
          hook_point: "input",
          enforcement_mode: "monitor",
          kind: "semantic",
          embedding_model: EMBED_MODEL,
          deny_examples: [prototype],
          deny_threshold: 0.99,
        },
        { attach: false },
      );
      await seed.attachGuardrailToModel(guardrail.id, overflowGuardrailModelID, 200 + index);
    }

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
    await streamUpstream?.close();
    await embed?.close();
    await sls?.close();
  });

  test("semantic routing records its real batched embedding on the chat parent", async () => {
    const { sls, embed } = requireRuntime();

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
    expect(row.has("gateway_embedding_calls_dropped")).toBe(false);
    expectUnchangedParentUsage(row);
    // The route target, not the detached embedding model, still owns the
    // parent event's pricing and attempt attribution.
    expect(row.get("model_id")).toBe(routeTargetModelID);
    expect(row.get("attempt_model")).toBe(ROUTE_TARGET);
    expect(row.get("attempt_model")).not.toBe(EMBED_MODEL);
  });

  test("a streamed semantic route keeps its terminal child audit on one parent", async () => {
    const { app, sls, embed, streamUpstream } = requireRuntime();

    const embeddingCallsBefore = embed.callCount();
    const upstreamCallsBefore = streamUpstream.receivedRequests.length;
    const response = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
      },
      body: JSON.stringify({
        model: STREAM_ROUTER_MODEL,
        messages: [{ role: "user", content: ROUTE_PROMPT }],
        stream: true,
      }),
    });
    const streamBody = await response.text();
    const requestId = response.headers.get("x-aisix-request-id") ?? "";

    expect(response.status).toBe(200);
    expect(response.headers.get("content-type")).toContain("text/event-stream");
    expect(response.headers.get("x-aisix-route")).toBe("route-topic");
    expect(response.headers.get("x-aisix-served-by")).toBe(STREAM_ROUTE_TARGET);
    expect(streamBody).toContain(STREAM_RESPONSE_TEXT);
    expect(streamBody).toContain("data: [DONE]");
    expect(requestId, "the DP stamps the streamed parent request id").not.toBe("");
    expect(embed.callCount()).toBe(embeddingCallsBefore + 1);

    // This proves the real DP streamed from the dedicated upstream rather
    // than merely turning a buffered semantic response into SSE locally.
    expect(streamUpstream.receivedRequests).toHaveLength(upstreamCallsBefore + 1);
    const upstreamRequest = streamUpstream.receivedRequests.at(-1)!;
    expect(upstreamRequest.path).toBe("/v1/chat/completions");
    expect(JSON.parse(upstreamRequest.body)).toMatchObject({
      stream: true,
      stream_options: { include_usage: true },
    });

    const row = await usageRow(requestId);
    expect(rowsForRequest(sls, requestId)).toHaveLength(1);
    const calls = embeddingCalls(row);
    expect(calls).toHaveLength(1);
    expectSucceededCall(calls[0]!, "semantic_route", embeddingModelID, 20);
    expect(row.has("gateway_embedding_calls_dropped")).toBe(false);

    // These are the streamed target's terminal usage values, not the
    // semantic embedding's values. Chat streaming intentionally leaves the
    // DP-provided cost at zero: the control plane prices it on ingestion.
    expect(row.get("prompt_tokens")).toBe(String(STREAM_USAGE.prompt_tokens));
    expect(row.get("completion_tokens")).toBe(String(STREAM_USAGE.completion_tokens));
    expect(row.get("total_tokens")).toBe(String(STREAM_USAGE.total_tokens));
    expect(Number(row.get("cost_usd"))).toBe(0);
    expect(streamRouteTargetModelID).not.toBe("");
    expect(row.get("model_id")).toBe(streamRouteTargetModelID);
    expect(row.get("attempt_model")).toBe(STREAM_ROUTE_TARGET);
    expect(row.get("attempt_model")).not.toBe(EMBED_MODEL);
  });

  test("missing provider embedding usage is explicit on the real parent event", async () => {
    const { sls, embed } = requireRuntime();

    const callsBefore = embed.callCount();
    const response = await chat(UNAVAILABLE_USAGE_ROUTER_MODEL, ROUTE_PROMPT);
    expect(response.status).toBe(200);
    expect(response.route).toBe("route-topic");
    expect(response.servedBy).toBe(ROUTE_TARGET);
    expect(embed.callCount()).toBe(callsBefore + 1);

    const row = await usageRow(response.requestId);
    expect(rowsForRequest(sls, response.requestId)).toHaveLength(1);
    const calls = embeddingCalls(row);
    expect(calls).toHaveLength(1);
    const call = calls[0]!;
    expect(call.count).toBe(1);
    expect(call.purpose).toBe("semantic_route");
    expect(call.embedding_model_id).toBe(unavailableUsageEmbeddingModelID);
    expect(call.outcome).toBe("succeeded");
    expect(call.usage_source).toBe("unavailable");
    expect(call.prompt_tokens).toBe(0);
    expect(call.total_tokens).toBe(0);
    expect(call.latency_ms).toBeGreaterThan(0);
    expect(row.has("gateway_embedding_calls_dropped")).toBe(false);
    expectUnchangedParentUsage(row);
  });

  test("a semantic embedding bridge error retains its failed child on the 503 parent", async () => {
    const { sls, embed, upstream } = requireRuntime();

    const failedEmbeddingCallsBefore = embed.callCountFor(FAILED_USAGE_UPSTREAM_MODEL);
    const upstreamCallsBefore = upstream.receivedRequests.length;
    const response = await chat(FAILED_USAGE_ROUTER_MODEL, ROUTE_PROMPT);
    expect(response.status).toBe(503);
    expect(response.requestId, "the DP stamps the failed parent request id").not.toBe("");
    expect(embed.callCountFor(FAILED_USAGE_UPSTREAM_MODEL)).toBe(failedEmbeddingCallsBefore + 1);
    expect(
      upstream.receivedRequests.length,
      "on_embedding_failure=fail must not dispatch a chat target",
    ).toBe(upstreamCallsBefore);

    const row = await usageRow(response.requestId);
    expect(rowsForRequest(sls, response.requestId)).toHaveLength(1);
    expect(row.get("status_code")).toBe("503");

    const calls = embeddingCalls(row);
    expect(calls).toHaveLength(1);
    expect(row.has("gateway_embedding_calls_dropped")).toBe(false);
    const call = calls[0]!;
    expect(call.count).toBe(1);
    expect(call.purpose).toBe("semantic_route");
    expect(call.embedding_model_id).toBe(failedUsageEmbeddingModelID);
    expect(call.prompt_tokens).toBe(0);
    expect(call.total_tokens).toBe(0);
    expect(call.usage_source).toBe("unavailable");
    expect(call.outcome).toBe("failed");

    // The bridge selected no target, so its zero usage is child-only. The
    // parent identifies the resolved semantic router, never the embedding or
    // a direct target attempt.
    expect(row.get("prompt_tokens") ?? "0").toBe("0");
    expect(row.get("completion_tokens") ?? "0").toBe("0");
    expect(row.get("total_tokens") ?? "0").toBe("0");
    expect(Number(row.get("cost_usd") ?? "0")).toBe(0);
    expect(row.get("requested_model")).toBe(FAILED_USAGE_ROUTER_MODEL);
    expect(failedUsageRouterModelID).not.toBe("");
    expect(row.get("model_id")).toBe(failedUsageRouterModelID);
    expect(row.get("attempt_model") ?? "").toBe("");
    expect(row.get("model_id") ?? "").not.toBe(failedUsageEmbeddingModelID);
    expect(row.get("attempt_model") ?? "").not.toBe(FAILED_USAGE_EMBED_MODEL);
  });

  test("a failed semantic embedding remains child work after its default target succeeds", async () => {
    const { sls, embed, upstream } = requireRuntime();

    const failedEmbeddingCallsBefore = embed.callCountFor(FAILED_USAGE_UPSTREAM_MODEL);
    const upstreamCallsBefore = upstream.receivedRequests.length;
    const response = await chat(FALLBACK_USAGE_ROUTER_MODEL, ROUTE_PROMPT);
    expect(response.status).toBe(200);
    expect(response.servedBy).toBe(ROUTE_DEFAULT);
    expect(embed.callCountFor(FAILED_USAGE_UPSTREAM_MODEL)).toBe(failedEmbeddingCallsBefore + 1);
    expect(upstream.receivedRequests.length).toBe(upstreamCallsBefore + 1);

    const row = await usageRow(response.requestId);
    expect(rowsForRequest(sls, response.requestId)).toHaveLength(1);
    expect(row.get("status_code")).toBe("200");
    const calls = embeddingCalls(row);
    expect(calls).toHaveLength(1);
    const call = calls[0]!;
    expect(call.count).toBe(1);
    expect(call.purpose).toBe("semantic_route");
    expect(call.embedding_model_id).toBe(failedUsageEmbeddingModelID);
    expect(call.prompt_tokens).toBe(0);
    expect(call.total_tokens).toBe(0);
    expect(call.usage_source).toBe("unavailable");
    expect(call.outcome).toBe("failed");
    expect(call.latency_ms).toBeGreaterThan(0);
    expect(row.has("gateway_embedding_calls_dropped")).toBe(false);
    expectUnchangedParentUsage(row);

    // The failed bridge belongs to the completed parent event, while target
    // attribution remains the direct fallback the client actually received.
    expect(row.get("requested_model")).toBe(FALLBACK_USAGE_ROUTER_MODEL);
    expect(routeDefaultModelID).not.toBe("");
    expect(row.get("model_id")).toBe(routeDefaultModelID);
    expect(row.get("attempt_model")).toBe(ROUTE_DEFAULT);
    expect(row.get("model_id")).not.toBe(failedUsageEmbeddingModelID);
    expect(row.get("attempt_model")).not.toBe(FAILED_USAGE_EMBED_MODEL);
  });

  test(
    "a client cancellation during semantic routing keeps its failed child on the terminal parent",
    async () => {
      const { app, sls, embed, upstream } = requireRuntime();

      const slowEmbeddingCallsBefore = embed.callCountFor(CANCELLED_USAGE_UPSTREAM_MODEL);
      const upstreamCallsBefore = upstream.receivedRequests.length;
      const controller = new AbortController();
      const inflight = fetch(`${app.proxyUrl}/v1/chat/completions`, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          authorization: `Bearer ${CALLER_PLAINTEXT}`,
        },
        body: JSON.stringify({
          model: CANCELLED_ROUTER_MODEL,
          messages: [{ role: "user", content: ROUTE_PROMPT }],
        }),
        signal: controller.signal,
      });

      // Abort only after the real DP has reached the slow embedding
      // provider. A fixed delay could fire before semantic routing begins,
      // in which case the resulting 499 would prove nothing about the
      // in-flight bridge guard.
      for (
        let i = 0;
        i < 200 && embed.callCountFor(CANCELLED_USAGE_UPSTREAM_MODEL) === slowEmbeddingCallsBefore;
        i++
      ) {
        await new Promise((resolve) => setTimeout(resolve, 25));
      }
      expect(
        embed.callCountFor(CANCELLED_USAGE_UPSTREAM_MODEL),
        "the DP never dispatched the slow semantic embedding",
      ).toBe(slowEmbeddingCallsBefore + 1);
      controller.abort();
      await expect(inflight).rejects.toThrow();

      const row = await waitForSlsLog(
        sls,
        LOGSTORE,
        (log) => log.get("requested_model") === CANCELLED_ROUTER_MODEL,
        `the terminal 499 UsageEvent for ${CANCELLED_ROUTER_MODEL}`,
        20_000,
      );
      const requestId = row.get("request_id") ?? "";
      expect(requestId, "the cancellation event retains its request id").not.toBe("");
      expect(
        upstream.receivedRequests.length,
        "semantic routing must not dispatch a target after the client cancellation",
      ).toBe(upstreamCallsBefore);

      // The direct-model barrier is later in the exporter FIFO queue, so by
      // the time it is visible every row the cancelled request could emit is
      // visible too. That lets this assert exactly one terminal parent row.
      const barrier = await chat(ROUTE_DEFAULT, SLS_BARRIER_PROMPT);
      expect(barrier.status, "the SLS barrier must complete").toBe(200);
      await waitForSlsLog(
        sls,
        LOGSTORE,
        (barrierRow) => barrierRow.get("request_id") === barrier.requestId,
        `the SLS barrier UsageEvent for ${barrier.requestId}`,
      );
      expect(rowsForRequest(sls, requestId)).toHaveLength(1);

      expect(row.get("status_code")).toBe("499");
      expect(row.get("error_class")).toBe("client_disconnected");
      expect(row.get("error_message")).toContain("before the response head");
      expect(row.get("operation")).toBe("chat");

      const calls = embeddingCalls(row);
      expect(calls).toHaveLength(1);
      expect(row.has("gateway_embedding_calls_dropped")).toBe(false);
      const call = calls[0]!;
      expect(call.count).toBe(1);
      expect(call.purpose).toBe("semantic_route");
      expect(call.embedding_model_id).toBe(cancelledEmbeddingModelID);
      expect(call.prompt_tokens).toBe(0);
      expect(call.total_tokens).toBe(0);
      expect(call.usage_source).toBe("unavailable");
      expect(call.outcome).toBe("failed");
      expect(call.latency_ms).toBeGreaterThan(0);

      // The embedding never selected a route target, so it is child work
      // only: no target request has reached the chat upstream, and the
      // parent remains unpriced and unattributed to the embedding model.
      expect(row.get("prompt_tokens") ?? "0").toBe("0");
      expect(row.get("completion_tokens") ?? "0").toBe("0");
      expect(row.get("total_tokens") ?? "0").toBe("0");
      expect(Number(row.get("cost_usd") ?? "0")).toBe(0);
      expect(row.get("requested_model")).toBe(CANCELLED_ROUTER_MODEL);
      expect(row.get("model_id") ?? "").toBe("");
      expect(row.get("attempt_model") ?? "").toBe("");
      expect(row.get("model_id") ?? "").not.toBe(cancelledEmbeddingModelID);
      expect(row.get("attempt_model") ?? "").not.toBe(CANCELLED_EMBED_MODEL);
    },
    60_000,
  );

  test("semantic guardrail records prototypes and candidate only on its parent", async () => {
    const { sls, embed } = requireRuntime();

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

  test("an output semantic guardrail keeps post-target child work on the routed parent", async () => {
    const { sls, embed } = requireRuntime();

    const callsBefore = embed.callCount();
    const response = await chat(OUTPUT_GUARDRAIL_ROUTER_MODEL, OUTPUT_ROUTE_PROMPT);
    expect(response.status).toBe(200);
    expect(response.route).toBe("output-route-topic");
    expect(response.servedBy).toBe(OUTPUT_GUARDRAIL_TARGET);
    // One routing bridge, then one prototype and one response-text bridge
    // after the direct target has completed.
    expect(embed.callCount()).toBe(callsBefore + 3);

    const row = await usageRow(response.requestId);
    expect(rowsForRequest(sls, response.requestId)).toHaveLength(1);
    const calls = embeddingCalls(row);
    expect(calls).toHaveLength(3);
    const routeCalls = calls.filter((call) => call.purpose === "semantic_route");
    const guardrailCalls = calls.filter((call) => call.purpose === "guardrail");
    expect(routeCalls).toHaveLength(1);
    expectSucceededCall(routeCalls[0]!, "semantic_route", embeddingModelID, 20);
    expect(guardrailCalls).toHaveLength(2);
    for (const call of guardrailCalls) {
      expectSucceededCall(call, "guardrail", embeddingModelID, 10);
    }
    expectUnchangedParentUsage(row);

    // The post-target child work cannot rewrite the semantic route's actual
    // target attribution.
    expect(outputGuardrailTargetModelID).not.toBe("");
    expect(row.get("model_id")).toBe(outputGuardrailTargetModelID);
    expect(row.get("attempt_model")).toBe(OUTPUT_GUARDRAIL_TARGET);
    expect(row.get("model_id")).not.toBe(embeddingModelID);
    expect(row.get("attempt_model")).not.toBe(EMBED_MODEL);
  });

  test("semantic guardrail child audit caps real bridge work without leaking text", async () => {
    const { sls, embed } = requireRuntime();

    const callsBefore = embed.callCount();
    const response = await chat(GUARDRAIL_OVERFLOW_MODEL, GUARDRAIL_OVERFLOW_PROMPT);
    expect(response.status).toBe(200);
    // One cold prototype bridge plus one request-data bridge for every
    // monitor-mode row. A block-mode short circuit cannot satisfy this.
    expect(embed.callCount()).toBe(callsBefore + GUARDRAIL_OVERFLOW_COUNT * 2);

    const row = await usageRow(response.requestId);
    expect(rowsForRequest(sls, response.requestId)).toHaveLength(1);
    const calls = embeddingCalls(row);
    expect(calls).toHaveLength(64);
    for (const call of calls) {
      expectSucceededCall(call, "guardrail", embeddingModelID, 10);
    }
    expect(row.get("gateway_embedding_calls_dropped")).toBe("2");
    expectUnchangedParentUsage(row);
    expect(row.get("model_id")).toBe(overflowGuardrailModelID);
    expect(row.get("requested_model")).toBe(GUARDRAIL_OVERFLOW_MODEL);
    expect(row.get("attempt_model")).toBeUndefined();

    // Metadata-only audit stays value-free: neither the caller's input nor
    // the operator's prototype texts may survive its bridge accounting.
    const audit = JSON.stringify(Object.fromEntries(row));
    for (const text of [GUARDRAIL_OVERFLOW_PROMPT, ...GUARDRAIL_OVERFLOW_PROTOTYPES]) {
      expect(audit).not.toContain(text);
    }
  });

  test("semantic-cache exact hit makes no embedding call and carries no child audit", async () => {
    const { sls, embed } = requireRuntime();

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
