import { createHash, randomUUID } from "node:crypto";
import { createServer, type IncomingHttpHeaders, type Server } from "node:http";
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
import { startMockOtlp, type MockOtlp } from "../harness/otlp-mock.js";
import { pickFreePort } from "../harness/ports.js";

// AISIX-Cloud#2075: the model calls the gateway makes on a request's behalf
// — a semantic router's embedding or classifier call, the semantic cache's
// lookup embedding, a semantic guardrail's embeddings — are each a CHILD
// usage event: same request id and caller as the request, the called
// model's own identity and tokens, `internal_call_purpose` set, never the
// request's terminal row. The request's own events are unchanged, and a
// semantic router's carry the target it selected (`route_target_model_id`).
//
// A real aisix DP + etcd, read through the SLS export it emits (one
// metadata-only logstore, one full-content logstore); the embedding,
// decision and chat upstreams are deterministic local endpoints.

const CALLER_PLAINTEXT = "sk-internal-call-usage-2075";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");

const CREDENTIAL_REF = "mock";
const SLS_PROJECT = "aisix-e2e-obs";
const LOGSTORE = "internal-call-usage-2075";
const FULL_LOGSTORE = "internal-call-usage-2075-full";
const EMBEDDING_DELAY_MS = 20;
// The cancellation case must leave the bridge pending long enough for the
// client to disconnect after the real DP has dispatched it.
const CANCELLED_EMBEDDING_DELAY_MS = 30_000;
const DECISION_SECRET = "ts-secret-2075";
const INLINE_DECISION_SECRET = "ts-inline-secret-2075";

const EMBED_MODEL = "icu-2075-embed";
const ROUTER_MODEL = "icu-2075-router";
const GROUP_ROUTER_MODEL = "icu-2075-group-router";
const ROUTE_TARGET = "icu-2075-route-target";
const ROUTE_GROUP = "icu-2075-route-group";
const ROUTE_GROUP_MEMBER = "icu-2075-route-group-member";
const ROUTE_DEFAULT = "icu-2075-route-default";
const STREAM_ROUTER_MODEL = "icu-2075-router-stream";
const STREAM_ROUTE_TARGET = "icu-2075-route-target-stream";
const GUARDRAIL_MODEL = "icu-2075-guardrail-chat";
const CUSTOM_TIMEOUT_MODEL = "icu-2075-custom-timeout-chat";
const CACHE_MODEL = "icu-2075-cache-chat";
const UNAVAILABLE_USAGE_EMBED_MODEL = "icu-2075-embed-usage-unavailable";
const UNAVAILABLE_USAGE_ROUTER_MODEL = "icu-2075-router-usage-unavailable";
const UNAVAILABLE_USAGE_UPSTREAM_MODEL = "embedding-usage-unavailable-mock";
const FAILED_EMBED_MODEL = "icu-2075-embed-failed";
const FAILED_ROUTER_MODEL = "icu-2075-router-failed";
const FAILED_UPSTREAM_MODEL = "embedding-usage-failed-mock";
const CANCELLED_EMBED_MODEL = "icu-2075-embed-cancelled";
const CANCELLED_ROUTER_MODEL = "icu-2075-router-cancelled";
const CANCELLED_UPSTREAM_MODEL = "embedding-usage-cancelled-mock";
const JEV_MODEL = "icu-2075-jev";
const JEV_UPSTREAM_MODEL = "jev-1.13.0";
const JEV_ROUTER_MODEL = "icu-2075-jev-router";
const JEV_INLINE_ROUTER_MODEL = "icu-2075-jev-inline-router";
const JEV_FAILED_ROUTER_MODEL = "icu-2075-jev-failed-router";

const ROUTE_EXAMPLE = "route-topic prototype";
// Distinct per router: prototype embeddings are cached across requests, and
// sharing one would make a later router's call depend on test order.
const GROUP_ROUTE_EXAMPLE = "group-route-topic prototype";
const STREAM_ROUTE_EXAMPLE = "stream-route-topic prototype";
const ROUTE_PROMPT = "route-topic caller question";
const STREAM_RESPONSE_TEXT = "streamed semantic route answer";
const STREAM_USAGE = { prompt_tokens: 13, completion_tokens: 5, total_tokens: 18 };
const chunk = (body: Record<string, unknown>) =>
  JSON.stringify({
    id: "cmpl-internal-call-stream",
    object: "chat.completion.chunk",
    created: 1_700_000_000,
    model: "gpt-4o-mini",
    ...body,
  });
const STREAM_EVENTS = [
  chunk({ choices: [{ index: 0, delta: { role: "assistant" }, finish_reason: null }] }),
  chunk({
    choices: [{ index: 0, delta: { content: STREAM_RESPONSE_TEXT }, finish_reason: null }],
  }),
  chunk({ choices: [{ index: 0, delta: {}, finish_reason: "stop" }] }),
  chunk({ choices: [], usage: STREAM_USAGE }),
  "[DONE]",
];
const GUARDRAIL_EXAMPLES = ["guardrail-prototype-first", "guardrail-prototype-second"];
const GUARDRAIL_PROMPT = "guardrail-candidate-allowed";
const CACHE_PROMPT = "cache-topic first wording";
const CACHE_PARAPHRASE = "cache-topic second wording";
const DECISION_PROMPT = "fix my python script";
const BARRIER_PROMPT = "internal-call-usage-sls-barrier";

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

/** OpenAI-compatible embedding endpoint: 10 prompt tokens per input, total = prompt + 1. */
async function startEmbeddingMock(): Promise<EmbeddingMock> {
  let calls = 0;
  const callsByModel = new Map<string, number>();
  const server: Server = createServer((req, res) => {
    res.on("error", () => {});
    let raw = "";
    req.on("data", (c: Buffer) => (raw += c.toString("utf8")));
    req.on("end", () => {
      const body = JSON.parse(raw || "{}") as { model?: string; input?: string | string[] };
      calls++;
      const model = body.model ?? "embedding-usage-mock";
      callsByModel.set(model, (callsByModel.get(model) ?? 0) + 1);
      const inputs = Array.isArray(body.input) ? body.input : [body.input ?? ""];
      const timer = setTimeout(
        () => {
          if (res.destroyed || res.writableEnded) return;
          res.setHeader("content-type", "application/json");
          if (model === FAILED_UPSTREAM_MODEL) {
            res.statusCode = 502;
            res.end(
              JSON.stringify({
                error: { message: `embedding upstream unavailable for ${inputs.join(" ")}` },
              }),
            );
            return;
          }
          const promptTokens = inputs.length * 10;
          res.statusCode = 200;
          res.end(
            JSON.stringify({
              object: "list",
              model,
              data: inputs.map((text, index) => ({
                object: "embedding",
                index,
                embedding: keywordVector(text),
              })),
              ...(model === UNAVAILABLE_USAGE_UPSTREAM_MODEL
                ? {}
                : { usage: { prompt_tokens: promptTokens, total_tokens: promptTokens + 1 } }),
            }),
          );
        },
        model === CANCELLED_UPSTREAM_MODEL ? CANCELLED_EMBEDDING_DELAY_MS : EMBEDDING_DELAY_MS,
      );
      res.once("close", () => clearTimeout(timer));
    });
  });
  const port = await pickFreePort();
  await new Promise<void>((resolve) => server.listen(port, "127.0.0.1", resolve));
  return {
    baseUrl: `http://127.0.0.1:${port}`,
    callCount: () => calls,
    callCountFor: (model) => callsByModel.get(model) ?? 0,
    async close() {
      await new Promise<void>((resolve, reject) =>
        server.close((err) => (err ? reject(err) : resolve())),
      );
    },
  };
}

interface DecisionCall {
  path: string;
  headers: IncomingHttpHeaders;
  body: Record<string, unknown> & { state?: string; model?: string };
}

/** A System One endpoint: picks `code` for "python", 500s for "boom". */
async function startDecisionMock(): Promise<{
  baseUrl: string;
  calls: DecisionCall[];
  close(): Promise<void>;
}> {
  const calls: DecisionCall[] = [];
  const server: Server = createServer((req, res) => {
    res.on("error", () => {});
    let raw = "";
    req.on("data", (c: Buffer) => (raw += c.toString("utf8")));
    req.on("end", () => {
      const body = JSON.parse(raw || "{}") as DecisionCall["body"];
      calls.push({ path: req.url ?? "", headers: req.headers, body });
      res.setHeader("content-type", "application/json");
      if ((body.state ?? "").includes("boom")) {
        res.statusCode = 500;
        res.end(JSON.stringify({ detail: `cannot decide ${body.state}` }));
        return;
      }
      res.statusCode = 200;
      res.end(
        JSON.stringify({
          id: "dec-2075",
          model: body.model,
          answers: { route: { type: "choice", choice: "code", confidence: 0.93 } },
          usage: { input_tokens: 300, output_tokens: 20 },
        }),
      );
    });
  });
  const port = await pickFreePort();
  await new Promise<void>((resolve) => server.listen(port, "127.0.0.1", resolve));
  return {
    baseUrl: `http://127.0.0.1:${port}`,
    calls,
    async close() {
      server.closeAllConnections();
      await new Promise<void>((resolve) => server.close(() => resolve()));
    },
  };
}

type Row = Map<string, string>;

interface ChatResponse {
  status: number;
  requestId: string;
  cache: string | null;
  cacheLayer: string | null;
  route: string | null;
  servedBy: string | null;
}

const isChild = (row: Row) => row.has("internal_call_purpose");

describe("internal model calls as child usage events (#2075)", () => {
  let app: SpawnedApp | undefined;
  let sls: MockSls | undefined;
  let otlp: MockOtlp | undefined;
  let embed: EmbeddingMock | undefined;
  let decisions: Awaited<ReturnType<typeof startDecisionMock>> | undefined;
  let upstream: OpenAiUpstream | undefined;
  let streamUpstream: OpenAiUpstream | undefined;
  let callerKeyID = "";
  const ids: Record<string, string> = {};

  function runtime() {
    if (!app || !sls || !embed || !decisions || !upstream || !streamUpstream) {
      throw new Error("internal call usage e2e setup did not complete");
    }
    return { app, sls, embed, decisions, upstream, streamUpstream };
  }

  async function chat(model: string, prompt: string): Promise<ChatResponse> {
    const res = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
      },
      body: JSON.stringify({ model, messages: [{ role: "user", content: prompt }] }),
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

  /** Every row of `requestId` in `logstore`, once all of them are exported. */
  async function rowsOf(requestId: string, logstore = LOGSTORE): Promise<Row[]> {
    expect(requestId, "the DP stamps the request id").not.toBe("");
    await waitForSlsLog(
      sls!,
      logstore,
      (row) => row.get("request_id") === requestId && !isChild(row),
      `the parent UsageEvent for ${requestId}`,
      20_000,
    );
    // The SLS writer is a FIFO queue: once a later request's row is
    // visible, every row of this one is too, so the counts below are exact.
    const barrier = await chat(ROUTE_DEFAULT, BARRIER_PROMPT);
    expect(barrier.status, "the SLS barrier must complete").toBe(200);
    await waitForSlsLog(
      sls!,
      logstore,
      (row) => row.get("request_id") === barrier.requestId,
      `the SLS barrier UsageEvent for ${barrier.requestId}`,
    );
    return slsLogsFor(sls!, logstore).filter((row) => row.get("request_id") === requestId);
  }

  /** The request's own rows: exactly one terminal row, and its children. */
  function split(rows: Row[]): { parent: Row; children: Row[] } {
    const own = rows.filter((row) => !isChild(row));
    const terminal = own.filter((row) => row.get("is_final_attempt") === "true");
    expect(terminal, "the request has exactly one terminal row").toHaveLength(1);
    return { parent: terminal[0]!, children: rows.filter(isChild) };
  }

  /** What every child shares with its request, and what it never is. */
  function expectChildOf(child: Row, parent: Row): void {
    expect(child.get("request_id")).toBe(parent.get("request_id"));
    expect(child.get("api_key_id")).toBe(callerKeyID);
    expect(child.get("api_key_id")).toBe(parent.get("api_key_id"));
    expect(child.get("inbound_protocol")).toBe(parent.get("inbound_protocol"));
    expect(child.get("is_final_attempt")).toBe("false");
    expect(child.has("route_target_model_id")).toBe(false);
    expect(child.has("cost_usd")).toBe(false);
  }

  function expectEmbeddingChild(
    child: Row,
    purpose: string,
    modelID: string,
    modelName: string,
    promptTokens: number,
  ): void {
    expect(child.get("internal_call_purpose")).toBe(purpose);
    expect(child.get("operation")).toBe("embeddings");
    expect(child.get("model_id")).toBe(modelID);
    expect(child.get("requested_model_id")).toBe(modelID);
    expect(child.get("requested_model")).toBe(modelName);
    expect(child.get("status_code")).toBe("200");
    // Exactly the provider's reported usage: 10 per input, total = prompt + 1.
    expect(child.get("prompt_tokens")).toBe(String(promptTokens));
    expect(child.get("total_tokens")).toBe(String(promptTokens + 1));
    // Every mock answer is delayed, so this is a real duration.
    expect(Number(child.get("upstream_latency_ms"))).toBeGreaterThan(0);
  }

  function expectParentUsage(row: Row): void {
    // The chat upstream's own usage; the embedding model adds nothing to it.
    expect(row.get("prompt_tokens")).toBe("11");
    expect(row.get("completion_tokens")).toBe("7");
    expect(row.get("total_tokens")).toBe("18");
  }

  beforeAll(async () => {
    const etcd = new EtcdClient();
    expect(await etcd.ping(), "internal call usage e2e requires etcd").toBe(true);

    sls = await startMockSls();
    otlp = await startMockOtlp();
    embed = await startEmbeddingMock();
    decisions = await startDecisionMock();
    upstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "cmpl-internal-call-usage",
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: "gpt-4o-mini",
        choices: [
          { index: 0, message: { role: "assistant", content: "upstream-answered" }, finish_reason: "stop" },
        ],
        usage: { prompt_tokens: 11, completion_tokens: 7, total_tokens: 18 },
      },
    });
    streamUpstream = await startOpenAiUpstream({ eventDelayMs: 2, streamEvents: STREAM_EVENTS });
    app = await spawnApp({
      exposeDebugHeaders: true,
      extraEnv: {
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: "mock-akid",
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: "mock-secret",
      },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);

    for (const [name, logstore, content_mode] of [
      ["internal-call-usage-sls", LOGSTORE, "metadata_only"],
      ["internal-call-usage-sls-full", FULL_LOGSTORE, "full"],
    ] as const) {
      await seed.createObservabilityExporter({
        name,
        enabled: true,
        kind: "aliyun_sls",
        endpoint: sls.url,
        project: SLS_PROJECT,
        logstore,
        credential_ref: CREDENTIAL_REF,
        content_mode,
      });
    }
    await seed.createObservabilityExporter({
      name: "internal-call-usage-otlp",
      enabled: true,
      kind: "otlp_http",
      endpoint: otlp.url,
    });

    const embeddingKey = await seed.createProviderKey({
      display_name: "icu-2075-embed-pk",
      api_key: "sk-embedding-mock",
      api_base: `${embed.baseUrl}/v1`,
    });
    for (const [name, upstreamModel] of [
      [EMBED_MODEL, "embedding-usage-mock"],
      [UNAVAILABLE_USAGE_EMBED_MODEL, UNAVAILABLE_USAGE_UPSTREAM_MODEL],
      [FAILED_EMBED_MODEL, FAILED_UPSTREAM_MODEL],
      [CANCELLED_EMBED_MODEL, CANCELLED_UPSTREAM_MODEL],
    ] as const) {
      const model = await seed.createModel({
        display_name: name,
        provider: "openai",
        model_name: upstreamModel,
        provider_key_id: embeddingKey.id,
        embedding: { dimensions: 4, normalize: true },
      });
      ids[name] = model.id;
    }

    const chatKey = await seed.createProviderKey({
      display_name: "icu-2075-chat-pk",
      api_key: "sk-chat-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    const streamKey = await seed.createProviderKey({
      display_name: "icu-2075-stream-pk",
      api_key: "sk-stream-mock",
      api_base: `${streamUpstream.baseUrl}/v1`,
    });
    for (const name of [
      ROUTE_TARGET,
      ROUTE_GROUP_MEMBER,
      ROUTE_DEFAULT,
      GUARDRAIL_MODEL,
      CUSTOM_TIMEOUT_MODEL,
      CACHE_MODEL,
    ]) {
      const model = await seed.createModel({
        display_name: name,
        provider: "openai",
        model_name: "gpt-4o-mini",
        provider_key_id: chatKey.id,
      });
      ids[name] = model.id;
    }
    const streamTarget = await seed.createModel({
      display_name: STREAM_ROUTE_TARGET,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: streamKey.id,
    });
    ids[STREAM_ROUTE_TARGET] = streamTarget.id;
    const group = await seed.createModel({
      display_name: ROUTE_GROUP,
      routing: { strategy: "failover", targets: [{ model: ROUTE_GROUP_MEMBER }] },
    });
    ids[ROUTE_GROUP] = group.id;

    const embeddingRouter = async (
      name: string,
      embeddingModel: string,
      target: string,
      example: string,
      extra: Record<string, unknown> = {},
    ) => {
      const model = await seed.createModel({
        display_name: name,
        semantic: {
          embedding_model: embeddingModel,
          routes: [{ name: "route-topic", target, examples: [example], threshold: 0.9 }],
          default: ROUTE_DEFAULT,
          match: { threshold: 0.9 },
          ...extra,
        },
      });
      ids[name] = model.id;
    };
    await embeddingRouter(ROUTER_MODEL, EMBED_MODEL, ROUTE_TARGET, ROUTE_EXAMPLE);
    await embeddingRouter(GROUP_ROUTER_MODEL, EMBED_MODEL, ROUTE_GROUP, GROUP_ROUTE_EXAMPLE);
    await embeddingRouter(STREAM_ROUTER_MODEL, EMBED_MODEL, STREAM_ROUTE_TARGET, STREAM_ROUTE_EXAMPLE);
    await embeddingRouter(
      UNAVAILABLE_USAGE_ROUTER_MODEL,
      UNAVAILABLE_USAGE_EMBED_MODEL,
      ROUTE_TARGET,
      ROUTE_EXAMPLE,
    );
    await embeddingRouter(FAILED_ROUTER_MODEL, FAILED_EMBED_MODEL, ROUTE_TARGET, ROUTE_EXAMPLE, {
      on_embedding_failure: "fail",
    });
    await embeddingRouter(CANCELLED_ROUTER_MODEL, CANCELLED_EMBED_MODEL, ROUTE_TARGET, ROUTE_EXAMPLE);

    // The classifier's direct model: its Provider Key's api_base ends in
    // `/v1`, and it carries a body override and a request header, all of
    // which the decision call must honour as a direct `/v1/systemone` does.
    const jevKey = await seed.createProviderKey({
      display_name: "icu-2075-jev-pk",
      api_key: DECISION_SECRET,
      provider: "typesafe",
      adapter: undefined,
      api_base: `${decisions.baseUrl}/v1`,
      request: {
        default_body_fields: { icu_override: "from-provider-key" },
        default_headers: { "x-icu-header": "key-${request.api_key.id}" },
      },
    });
    const jev = await seed.createModel({
      display_name: JEV_MODEL,
      provider: "typesafe",
      model_name: JEV_UPSTREAM_MODEL,
      provider_key_id: jevKey.id,
    });
    ids[JEV_MODEL] = jev.id;
    // An inline classifier's own key; its api_base ends in `/v1` too.
    const inlineKey = await seed.createProviderKey({
      display_name: "icu-2075-jev-inline-pk",
      api_key: INLINE_DECISION_SECRET,
      provider: "typesafe",
      adapter: undefined,
      api_base: `${decisions.baseUrl}/v1/`,
    });
    const notTypesafe = await seed.createProviderKey({
      display_name: "icu-2075-not-typesafe-pk",
      api_key: "sk-not-typesafe",
      api_base: decisions.baseUrl,
    });
    const jevRouter = async (name: string, classifier: Record<string, unknown>) => {
      const model = await seed.createModel({
        display_name: name,
        semantic: {
          classifier: { type: "jev", timeout_ms: 2_000, ...classifier },
          routes: [{ name: "code", target: ROUTE_TARGET, description: "programming questions" }],
          default: ROUTE_DEFAULT,
        },
      });
      ids[name] = model.id;
    };
    // Both spellings, as a control plane writes them for older gateways:
    // the inline pair names a key the classifier must NOT use.
    await jevRouter(JEV_ROUTER_MODEL, {
      model_id: jev.id,
      provider_key_id: notTypesafe.id,
      model: "jev-inline-must-not-be-sent",
    });
    await jevRouter(JEV_INLINE_ROUTER_MODEL, { provider_key_id: inlineKey.id, model: "jev-inline" });
    await jevRouter(JEV_FAILED_ROUTER_MODEL, { model_id: jev.id, on_failure: "fail" });

    const guardrail = await seed.createGuardrail(
      {
        enabled: true,
        name: "icu-2075-guardrail",
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
      guardrail_id: guardrail.id,
      scope_type: "model",
      scope_id: ids[GUARDRAIL_MODEL],
      priority: 100,
    });

    // A custom script whose embedding outlives the script's own time
    // budget: the gateway stops waiting, the client does not leave.
    const customTimeout = await seed.createGuardrail(
      {
        enabled: true,
        name: "icu-2075-custom-timeout",
        hook_point: "input",
        kind: "custom",
        fail_open: true,
        timeout_ms: 500,
        script: `export async function checkInput(ctx) {
  await aisix.embed("${CANCELLED_EMBED_MODEL}", [ctx.text]);
  return { action: "none" };
}`,
      },
      { attach: false },
    );
    await seed.attachGuardrailToModel(customTimeout.id, ids[CUSTOM_TIMEOUT_MODEL]!);

    await seed.createCachePolicy({
      name: "icu-2075-cache",
      backend: "memory",
      applies_to: `model:${CACHE_MODEL}`,
      ttl_seconds: 600,
      semantic: { embedding_model: EMBED_MODEL, threshold: 0.9 },
    });

    // Seeded last: its authenticating proves every row above has landed.
    const key = await seed.createApiKey({ key_hash: CALLER_KEY_HASH, allowed_models: ["*"] });
    callerKeyID = key.id;
    const probe = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    await waitConfigPropagation(async () => (await probe.listModels()).status === 200);
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    await streamUpstream?.close();
    await embed?.close();
    await decisions?.close();
    await sls?.close();
    await otlp?.close();
  });

  test("a semantic route's embedding is a child; the parent names the selected target", async () => {
    const { embed } = runtime();
    const callsBefore = embed.callCount();
    const response = await chat(ROUTER_MODEL, ROUTE_PROMPT);
    expect(response.status).toBe(200);
    expect(response.servedBy).toBe(ROUTE_TARGET);
    expect(embed.callCount()).toBe(callsBefore + 1);

    const { parent, children } = split(await rowsOf(response.requestId));
    expect(children).toHaveLength(1);
    const child = children[0]!;
    expectChildOf(child, parent);
    // One batch: the prompt and the cold route example.
    expectEmbeddingChild(child, "semantic_route_embedding", ids[EMBED_MODEL]!, EMBED_MODEL, 20);

    expectParentUsage(parent);
    expect(parent.get("requested_model")).toBe(ROUTER_MODEL);
    expect(parent.get("requested_model_id")).toBe(ids[ROUTER_MODEL]);
    expect(parent.get("model_id")).toBe(ids[ROUTE_TARGET]);
    expect(parent.get("route_target_model_id")).toBe(ids[ROUTE_TARGET]);
    expect(parent.has("gateway_embedding_calls")).toBe(false);
  });

  test("a route to a model group names the group as the target and its member as model_id", async () => {
    const response = await chat(GROUP_ROUTER_MODEL, ROUTE_PROMPT);
    expect(response.status).toBe(200);
    const { parent, children } = split(await rowsOf(response.requestId));
    expect(parent.get("route_target_model_id")).toBe(ids[ROUTE_GROUP]);
    expect(parent.get("model_id")).toBe(ids[ROUTE_GROUP_MEMBER]);
    expect(children.map((c) => c.get("internal_call_purpose"))).toEqual(["semantic_route_embedding"]);
  });

  test("a streamed semantic route has one terminal row and its embedding child", async () => {
    const { app, embed } = runtime();
    const callsBefore = embed.callCount();
    const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json", authorization: `Bearer ${CALLER_PLAINTEXT}` },
      body: JSON.stringify({
        model: STREAM_ROUTER_MODEL,
        messages: [{ role: "user", content: ROUTE_PROMPT }],
        stream: true,
      }),
    });
    const body = await res.text();
    expect(res.status).toBe(200);
    expect(body).toContain(STREAM_RESPONSE_TEXT);
    expect(embed.callCount()).toBe(callsBefore + 1);

    const { parent, children } = split(await rowsOf(res.headers.get("x-aisix-request-id") ?? ""));
    expect(children).toHaveLength(1);
    expectChildOf(children[0]!, parent);
    expectEmbeddingChild(children[0]!, "semantic_route_embedding", ids[EMBED_MODEL]!, EMBED_MODEL, 20);
    expect(parent.get("prompt_tokens")).toBe(String(STREAM_USAGE.prompt_tokens));
    expect(parent.get("model_id")).toBe(ids[STREAM_ROUTE_TARGET]);
    expect(parent.get("route_target_model_id")).toBe(ids[STREAM_ROUTE_TARGET]);
  });

  test("an embedding answer without usage is counted locally and flagged estimated", async () => {
    const response = await chat(UNAVAILABLE_USAGE_ROUTER_MODEL, ROUTE_PROMPT);
    expect(response.status).toBe(200);
    const { parent, children } = split(await rowsOf(response.requestId));
    expect(children).toHaveLength(1);
    const child = children[0]!;
    expectChildOf(child, parent);
    expect(child.get("model_id")).toBe(ids[UNAVAILABLE_USAGE_EMBED_MODEL]);
    expect(child.get("status_code")).toBe("200");
    expect(Number(child.get("prompt_tokens") ?? "0")).toBeGreaterThan(0);
    expect(child.get("usage_estimated")).toBe("true");
    expectParentUsage(parent);
  });

  test("an embedding answer with usage is not flagged estimated", async () => {
    const response = await chat(ROUTER_MODEL, ROUTE_PROMPT);
    const { children } = split(await rowsOf(response.requestId));
    expect(children).toHaveLength(1);
    expect(children[0]!.has("usage_estimated")).toBe(false);
  });

  test("a child's OTLP span is in the request's trace, under its SERVER span", async () => {
    const response = await chat(ROUTER_MODEL, ROUTE_PROMPT);
    expect(response.status).toBe(200);
    const childSpan = () =>
      otlp!.spans.find(
        (span) =>
          span.attributes["aisix.request_id"] === response.requestId &&
          span.attributes["aisix.internal_call_purpose"] === "semantic_route_embedding",
      );
    const parentSpan = () =>
      otlp!.spans.find(
        (span) =>
          span.attributes["aisix.request_id"] === response.requestId &&
          span.attributes["aisix.internal_call_purpose"] === undefined,
      );
    for (let i = 0; i < 100 && !(childSpan() && parentSpan()); i++) {
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
    const child = childSpan();
    const parent = parentSpan();
    expect(child, "the child's span").toBeDefined();
    expect(parent, "a span of the request itself").toBeDefined();
    expect(child!.kind).toBe(3);
    expect(child!.traceId).toBe(parent!.traceId);
    const server = otlp!.spans.find((span) => span.spanId === child!.parentSpanId);
    expect(server, "the child's parent span was exported").toBeDefined();
    expect(server!.kind).toBe(2);
    expect(server!.traceId).toBe(parent!.traceId);
    expect(BigInt(child!.startTimeUnixNano)).toBeGreaterThanOrEqual(
      BigInt(server!.startTimeUnixNano),
    );
    expect(BigInt(child!.endTimeUnixNano)).toBeLessThanOrEqual(BigInt(server!.endTimeUnixNano));
  });

  test("an embedding the gateway stopped waiting for is a timeout child, not a client disconnect", async () => {
    const { embed } = runtime();
    const slowBefore = embed.callCountFor(CANCELLED_UPSTREAM_MODEL);
    const response = await chat(CUSTOM_TIMEOUT_MODEL, "custom-timeout candidate");
    // The script's budget elapsed; the row fails open and the request is served.
    expect(response.status).toBe(200);
    expect(embed.callCountFor(CANCELLED_UPSTREAM_MODEL)).toBe(slowBefore + 1);
    const { parent, children } = split(await rowsOf(response.requestId));
    expect(parent.get("status_code")).toBe("200");
    expect(children).toHaveLength(1);
    const child = children[0]!;
    expectChildOf(child, parent);
    expect(child.get("internal_call_purpose")).toBe("guardrail_embedding");
    expect(child.get("model_id")).toBe(ids[CANCELLED_EMBED_MODEL]);
    expect(child.get("status_code")).toBe("504");
    expect(child.get("error_class")).toBe("timeout");
    expect(Number(child.get("upstream_latency_ms"))).toBeGreaterThan(0);
  });

  test("a failed embedding is a failed child that carries no captured content", async () => {
    const { upstream } = runtime();
    const upstreamBefore = upstream.receivedRequests.length;
    const response = await chat(FAILED_ROUTER_MODEL, ROUTE_PROMPT);
    expect(response.status).toBe(503);
    expect(upstream.receivedRequests.length, "on_embedding_failure=fail dispatches nothing").toBe(
      upstreamBefore,
    );

    for (const logstore of [LOGSTORE, FULL_LOGSTORE]) {
      const { parent, children } = split(await rowsOf(response.requestId, logstore));
      expect(children).toHaveLength(1);
      const child = children[0]!;
      expectChildOf(child, parent);
      expect(child.get("internal_call_purpose")).toBe("semantic_route_embedding");
      expect(child.get("model_id")).toBe(ids[FAILED_EMBED_MODEL]);
      expect(child.get("status_code")).toBe("502");
      expect(child.get("error_class")).toBe("upstream_status");
      expect(child.get("error_message") ?? "").not.toBe("");
      expect(child.get("prompt_tokens") ?? "0").toBe("0");
      expect(parent.get("status_code")).toBe("503");
      expect(parent.has("route_target_model_id")).toBe(false);
      // The embedded text is the caller's prompt; the upstream quoted it in
      // its error. Neither reaches the child, on either export.
      const text = JSON.stringify(Object.fromEntries(child));
      expect(text).not.toContain(ROUTE_PROMPT);
      for (const content of ["input_messages", "output_messages", "prompt", "response"]) {
        expect(child.has(content), `child carries ${content}`).toBe(false);
      }
      if (logstore === FULL_LOGSTORE) {
        // The full-content export does capture the PARENT failure row, so
        // the child's absence is not an exporter that captures nothing.
        expect(parent.get("input_messages") ?? parent.get("prompt") ?? "").toContain(ROUTE_PROMPT);
      }
    }
  });

  test(
    "a client cancellation during the embedding leaves a 499 child beside the one terminal row",
    async () => {
      const { app, sls, embed, upstream } = runtime();
      const slowBefore = embed.callCountFor(CANCELLED_UPSTREAM_MODEL);
      const upstreamBefore = upstream.receivedRequests.length;
      const controller = new AbortController();
      const inflight = fetch(`${app.proxyUrl}/v1/chat/completions`, {
        method: "POST",
        headers: { "content-type": "application/json", authorization: `Bearer ${CALLER_PLAINTEXT}` },
        body: JSON.stringify({
          model: CANCELLED_ROUTER_MODEL,
          messages: [{ role: "user", content: ROUTE_PROMPT }],
        }),
        signal: controller.signal,
      });
      // Abort only once the DP has reached the slow embedding provider.
      for (let i = 0; i < 200 && embed.callCountFor(CANCELLED_UPSTREAM_MODEL) === slowBefore; i++) {
        await new Promise((resolve) => setTimeout(resolve, 25));
      }
      expect(embed.callCountFor(CANCELLED_UPSTREAM_MODEL)).toBe(slowBefore + 1);
      controller.abort();
      await expect(inflight).rejects.toThrow();

      const terminal = await waitForSlsLog(
        sls,
        LOGSTORE,
        (row) => row.get("requested_model") === CANCELLED_ROUTER_MODEL && !isChild(row),
        `the terminal 499 UsageEvent for ${CANCELLED_ROUTER_MODEL}`,
        20_000,
      );
      expect(
        upstream.receivedRequests.length,
        "semantic routing dispatches no target after the cancellation",
      ).toBe(upstreamBefore);
      const { parent, children } = split(await rowsOf(terminal.get("request_id") ?? ""));
      expect(parent.get("status_code")).toBe("499");
      expect(children).toHaveLength(1);
      const child = children[0]!;
      expectChildOf(child, parent);
      expect(child.get("internal_call_purpose")).toBe("semantic_route_embedding");
      expect(child.get("model_id")).toBe(ids[CANCELLED_EMBED_MODEL]);
      expect(child.get("status_code")).toBe("499");
      expect(child.get("error_class")).toBe("client_disconnected");
      expect(Number(child.get("upstream_latency_ms"))).toBeGreaterThan(0);
    },
    60_000,
  );

  test("a classifier naming a direct model calls it as a direct /v1/systemone does", async () => {
    const { decisions } = runtime();
    const before = decisions.calls.length;
    const response = await chat(JEV_ROUTER_MODEL, DECISION_PROMPT);
    expect(response.status).toBe(200);
    expect(response.route).toBe("code");
    expect(decisions.calls.length).toBe(before + 1);

    const call = decisions.calls.at(-1)!;
    // The key's api_base ends in `/v1`: one version segment, not two.
    expect(call.path).toBe("/v1/systemone");
    expect(call.headers.authorization).toBe(`Bearer ${DECISION_SECRET}`);
    expect(call.headers["x-aisix-request-id"]).toBe(response.requestId);
    expect(call.headers["x-icu-header"]).toBe(`key-${callerKeyID}`);
    expect(call.body.model).toBe(JEV_UPSTREAM_MODEL);
    expect(call.body.icu_override).toBe("from-provider-key");
    expect(call.body.state).toBe(DECISION_PROMPT);

    const { parent, children } = split(await rowsOf(response.requestId));
    expect(children).toHaveLength(1);
    const child = children[0]!;
    expectChildOf(child, parent);
    expect(child.get("internal_call_purpose")).toBe("semantic_route_decision");
    expect(child.get("operation")).toBe("systemone");
    expect(child.get("model_id")).toBe(ids[JEV_MODEL]);
    expect(child.get("requested_model_id")).toBe(ids[JEV_MODEL]);
    expect(child.get("requested_model")).toBe(JEV_MODEL);
    expect(child.get("status_code")).toBe("200");
    expect(child.get("prompt_tokens")).toBe("300");
    expect(child.get("completion_tokens")).toBe("20");
    expect(parent.get("model_id")).toBe(ids[ROUTE_TARGET]);
    expect(parent.get("route_target_model_id")).toBe(ids[ROUTE_TARGET]);
  });

  test("an inline classifier's child names no model; its api_base `/v1` is not doubled", async () => {
    const { decisions } = runtime();
    const before = decisions.calls.length;
    const response = await chat(JEV_INLINE_ROUTER_MODEL, DECISION_PROMPT);
    expect(response.status).toBe(200);
    expect(response.route).toBe("code");
    expect(decisions.calls.length).toBe(before + 1);
    const call = decisions.calls.at(-1)!;
    expect(call.path).toBe("/v1/systemone");
    expect(call.headers.authorization).toBe(`Bearer ${INLINE_DECISION_SECRET}`);
    expect(call.body.model).toBe("jev-inline");

    const { parent, children } = split(await rowsOf(response.requestId));
    expect(children).toHaveLength(1);
    const child = children[0]!;
    expectChildOf(child, parent);
    expect(child.get("internal_call_purpose")).toBe("semantic_route_decision");
    expect(child.get("operation")).toBe("systemone");
    expect(child.has("model_id")).toBe(false);
    expect(child.has("requested_model_id")).toBe(false);
    expect(child.get("prompt_tokens")).toBe("300");
    expect(child.get("completion_tokens")).toBe("20");
  });

  test("a failed classifier call is a failed child without captured content", async () => {
    const response = await chat(JEV_FAILED_ROUTER_MODEL, "boom python");
    expect(response.status).toBe(503);
    for (const logstore of [LOGSTORE, FULL_LOGSTORE]) {
      const { parent, children } = split(await rowsOf(response.requestId, logstore));
      expect(children).toHaveLength(1);
      const child = children[0]!;
      expectChildOf(child, parent);
      expect(child.get("internal_call_purpose")).toBe("semantic_route_decision");
      expect(child.get("model_id")).toBe(ids[JEV_MODEL]);
      expect(child.get("status_code")).toBe("500");
      expect(child.get("error_class")).toBe("upstream_status");
      expect(JSON.stringify(Object.fromEntries(child))).not.toContain("boom");
      for (const content of ["input_messages", "output_messages", "prompt", "response"]) {
        expect(child.has(content), `child carries ${content}`).toBe(false);
      }
    }
  });

  test("a semantic guardrail's embeddings are guardrail children", async () => {
    const { embed } = runtime();
    const callsBefore = embed.callCount();
    const response = await chat(GUARDRAIL_MODEL, GUARDRAIL_PROMPT);
    expect(response.status).toBe(200);
    // A cold guardrail embeds its prototypes once and the candidate once.
    expect(embed.callCount()).toBe(callsBefore + 2);

    const { parent, children } = split(await rowsOf(response.requestId));
    expect(children).toHaveLength(2);
    const byTokens = [...children].sort(
      (a, b) => Number(a.get("prompt_tokens")) - Number(b.get("prompt_tokens")),
    );
    expectEmbeddingChild(byTokens[0]!, "guardrail_embedding", ids[EMBED_MODEL]!, EMBED_MODEL, 10);
    expectEmbeddingChild(byTokens[1]!, "guardrail_embedding", ids[EMBED_MODEL]!, EMBED_MODEL, 20);
    for (const child of children) expectChildOf(child, parent);
    expectParentUsage(parent);
    expect(parent.get("model_id")).toBe(ids[GUARDRAIL_MODEL]);
    expect(parent.has("route_target_model_id")).toBe(false);
  });

  test("a semantic-cache hit still has the child of the embedding that found it", async () => {
    const { embed } = runtime();
    const miss = await chat(CACHE_MODEL, CACHE_PROMPT);
    expect(miss.status).toBe(200);
    expect(miss.cache).toBe("miss");
    const missRows = split(await rowsOf(miss.requestId));
    expect(missRows.children).toHaveLength(1);
    expectEmbeddingChild(
      missRows.children[0]!,
      "semantic_cache_embedding",
      ids[EMBED_MODEL]!,
      EMBED_MODEL,
      10,
    );

    const callsBefore = embed.callCount();
    const hit = await chat(CACHE_MODEL, CACHE_PARAPHRASE);
    expect(hit.status).toBe(200);
    expect(hit.cache).toBe("hit");
    expect(hit.cacheLayer).toBe("semantic");
    expect(embed.callCount()).toBe(callsBefore + 1);
    const { parent, children } = split(await rowsOf(hit.requestId));
    expect(children).toHaveLength(1);
    expectChildOf(children[0]!, parent);
    expectEmbeddingChild(children[0]!, "semantic_cache_embedding", ids[EMBED_MODEL]!, EMBED_MODEL, 10);

    // An exact hit is answered before any embedding is made: no child.
    const exactBefore = embed.callCount();
    const exact = await chat(CACHE_MODEL, CACHE_PROMPT);
    expect(exact.cache).toBe("hit");
    expect(exact.cacheLayer).toBe("exact");
    expect(embed.callCount()).toBe(exactBefore);
    expect(split(await rowsOf(exact.requestId)).children).toHaveLength(0);
  });

  test("a request no internal call served has no children and no route target", async () => {
    const response = await chat(ROUTE_DEFAULT, "a plain request");
    expect(response.status).toBe(200);
    const { parent, children } = split(await rowsOf(response.requestId));
    expect(children).toHaveLength(0);
    expect(parent.has("route_target_model_id")).toBe(false);
    expect(parent.has("internal_call_purpose")).toBe(false);
  });
});
