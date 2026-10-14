import { createHash } from "node:crypto";
import { createServer, type Server } from "node:http";
import { gunzipSync } from "node:zlib";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  pickFreePort,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";
import {
  startMockOtlp,
  type CapturedSpan,
  type MockOtlp,
} from "../harness/otlp-mock.js";

// E2E: the built-in OTLP exporter's spans follow the released OpenTelemetry
// GenAI semantic conventions (semantic-conventions v1.41.1) and the HTTP
// server span conventions, observed on the raw OTLP/HTTP payload a real
// trace backend receives.
//
// - The inference span is named `{gen_ai.operation.name} {gen_ai.request.model}`,
//   where the model is the one the gateway asked the upstream for — the
//   caller's alias is `aisix.requested_model` — and carries
//   `gen_ai.provider.name`, `server.address` / `server.port`, the caller's
//   request parameters, and usage under the conventions' accounting:
//   `gen_ai.usage.input_tokens` includes every cached token whether the
//   upstream reported its cache inside the prompt count (OpenAI) or beside
//   it (Anthropic), with `gen_ai.usage.cache_read.input_tokens` /
//   `gen_ai.usage.cache_creation.input_tokens` beside it.
// - Success leaves span status unset; a failure sets ERROR with `error.type`
//   on the CLIENT spans, while the SERVER span reports only a 5xx as its
//   own error.
// - The SERVER span is `{method} {route}` with the HTTP server attributes;
//   only the attempt span reads as a GenAI operation.
// - `content_mode: full` exports the captured exchange as
//   `gen_ai.input.messages` / `gen_ai.output.messages` /
//   `gen_ai.system_instructions`.
// - The Datadog exporter's `gen_ai.usage.input_tokens` follows the same
//   accounting.

const CALLER = "sk-otlp-genai-semconv-caller";
const CALLER_HASH = createHash("sha256").update(CALLER).digest("hex");
const CREDENTIAL_REF = "e2e";
const DD_API_KEY = "dd-genai-semconv-key";
const INTAKE_PATH = "/api/v2/logs";

const OPENAI_MODEL = "semconv-openai";
const OPENAI_UPSTREAM_MODEL = "gpt-4o-mini";
const ANTHROPIC_MODEL = "semconv-anthropic";
const ANTHROPIC_UPSTREAM_MODEL = "claude-3-5-haiku-20241022";
const EMBED_MODEL = "semconv-embed";
const EMBED_UPSTREAM_MODEL = "text-embedding-3-small";
const FAILING_MODEL = "semconv-failing";
const REFUSING_MODEL = "semconv-refusing";
const STREAM_MODEL = "semconv-stream";

/** OTLP SpanKind values (opentelemetry-proto trace.proto). */
const KIND_SERVER = 2;
const KIND_CLIENT = 3;
const STATUS_UNSET = 0;
const STATUS_ERROR = 2;

const OPENAI_BODY = {
  id: "chatcmpl-semconv",
  object: "chat.completion",
  created: 1_700_000_000,
  model: "gpt-4o-mini-2024-07-18",
  choices: [
    { index: 0, message: { role: "assistant", content: "semconv reply" }, finish_reason: "stop" },
  ],
  usage: {
    prompt_tokens: 120,
    completion_tokens: 30,
    total_tokens: 150,
    prompt_tokens_details: { cached_tokens: 100, cache_write_tokens: 20 },
    completion_tokens_details: { reasoning_tokens: 12 },
  },
};

const ANTHROPIC_BODY = {
  id: "msg_semconv",
  type: "message",
  role: "assistant",
  content: [{ type: "text", text: "anthropic reply" }],
  model: ANTHROPIC_UPSTREAM_MODEL,
  stop_reason: "end_turn",
  usage: {
    input_tokens: 10,
    output_tokens: 5,
    cache_read_input_tokens: 300,
    cache_creation_input_tokens: 40,
  },
};

const EMBED_BODY = {
  object: "list",
  data: [{ object: "embedding", index: 0, embedding: [0.1, 0.2, 0.3] }],
  model: EMBED_UPSTREAM_MODEL,
  usage: { prompt_tokens: 4, total_tokens: 4 },
};

function streamChunk(json: Record<string, unknown>): string {
  return JSON.stringify({
    id: "chatcmpl-semconv-stream",
    object: "chat.completion.chunk",
    created: 1_700_000_000,
    model: OPENAI_UPSTREAM_MODEL,
    ...json,
  });
}

const STREAM_EVENTS = [
  streamChunk({ choices: [{ index: 0, delta: { role: "assistant" }, finish_reason: null }] }),
  streamChunk({ choices: [{ index: 0, delta: { content: "streamed" }, finish_reason: null }] }),
  streamChunk({ choices: [{ index: 0, delta: {}, finish_reason: "stop" }] }),
  streamChunk({ choices: [], usage: { prompt_tokens: 6, completion_tokens: 2, total_tokens: 8 } }),
  "[DONE]",
];

interface MockDatadog {
  site: string;
  logs: Record<string, unknown>[];
  close(): Promise<void>;
}

async function startMockDatadog(): Promise<MockDatadog> {
  const logs: Record<string, unknown>[] = [];
  const server: Server = createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on("data", (c: Buffer) => chunks.push(c));
    req.on("end", () => {
      if (req.method === "POST" && (req.url ?? "").split("?")[0] === INTAKE_PATH) {
        try {
          const parsed: unknown = JSON.parse(gunzipSync(Buffer.concat(chunks)).toString("utf8"));
          if (Array.isArray(parsed)) logs.push(...(parsed as Record<string, unknown>[]));
        } catch {
          // An undecodable body leaves nothing captured; the wait below fails.
        }
      }
      res.statusCode = 202;
      res.end();
    });
  });
  const port = await pickFreePort();
  await new Promise<void>((resolve) => server.listen(port, "127.0.0.1", resolve));
  return {
    site: `127.0.0.1:${port}`,
    logs,
    async close() {
      await new Promise<void>((resolve, reject) => server.close((e) => (e ? reject(e) : resolve())));
    },
  };
}

async function waitFor<T>(what: string, probe: () => T | undefined, timeoutMs = 15_000): Promise<T> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const hit = probe();
    if (hit !== undefined) return hit;
    await new Promise((r) => setTimeout(r, 50));
  }
  throw new Error(`timed out waiting for ${what}`);
}

/**
 * The request's spans once its SERVER span arrived. The terminal event
 * carries SERVER, logical and attempt spans in one record, so the SERVER
 * span's arrival means its whole emission landed.
 */
async function requestSpans(otlp: MockOtlp, requestId: string): Promise<CapturedSpan[]> {
  await waitFor(`SERVER span of ${requestId}`, () =>
    otlp.spans.find((s) => s.attributes["aisix.request_id"] === requestId && s.kind === KIND_SERVER),
  );
  return otlp.spans.filter((s) => s.attributes["aisix.request_id"] === requestId);
}

/** The GenAI operation span of a single-attempt request. */
function attemptSpan(spans: CapturedSpan[]): CapturedSpan {
  const server = spans.find((s) => s.kind === KIND_SERVER)!;
  const logical = spans.find((s) => s.kind === KIND_CLIENT && s.parentSpanId === server.spanId)!;
  const attempt = spans.find((s) => s.kind === KIND_CLIENT && s.parentSpanId === logical.spanId);
  expect(attempt, "an attempt span under the logical span").toBeTruthy();
  return attempt!;
}

function portOf(baseUrl: string): number {
  return Number(new URL(baseUrl).port);
}

describe("OTLP spans follow the GenAI semantic conventions", () => {
  let etcdReachable = false;
  let app: SpawnedApp | undefined;
  let otlp: MockOtlp | undefined;
  let dd: MockDatadog | undefined;
  const upstreams: Record<string, OpenAiUpstream> = {};

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstreams.openai = await startOpenAiUpstream({ nonStreamBody: OPENAI_BODY });
    upstreams.anthropic = await startOpenAiUpstream({ nonStreamBody: ANTHROPIC_BODY });
    upstreams.embed = await startOpenAiUpstream({ nonStreamBody: EMBED_BODY });
    upstreams.failing = await startOpenAiUpstream({
      status: 500,
      errorBody: { error: { message: "upstream exploded", type: "server_error" } },
    });
    upstreams.refusing = await startOpenAiUpstream({
      status: 400,
      errorBody: { error: { message: "bad request", type: "invalid_request_error" } },
    });
    upstreams.stream = await startOpenAiUpstream({
      streamEvents: STREAM_EVENTS,
      firstEventDelayMs: 60,
    });
    otlp = await startMockOtlp();
    dd = await startMockDatadog();

    app = await spawnApp({
      extraEnv: { [`DD_CRED_${CREDENTIAL_REF.toUpperCase()}_API_KEY`]: DD_API_KEY },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createObservabilityExporter({
      name: "semconv-otlp",
      enabled: true,
      kind: "otlp_http",
      endpoint: otlp.url,
      content_mode: "full",
      content_max_bytes: 8192,
    });
    await seed.createObservabilityExporter({
      name: "semconv-datadog",
      enabled: true,
      kind: "datadog",
      site: dd.site,
      credential_ref: CREDENTIAL_REF,
      service: "semconv-e2e",
      content_mode: "metadata_only",
    });

    const openaiModel = async (name: string, upstream: OpenAiUpstream, modelName: string) => {
      const pk = await seed.createProviderKey({
        display_name: `${name}-pk`,
        provider: "openai",
        secret: "sk-mock",
        api_base: `${upstream.baseUrl}/v1`,
      });
      await seed.createModel({
        display_name: name,
        provider: "openai",
        model_name: modelName,
        provider_key_id: pk.id,
      });
    };
    await openaiModel(OPENAI_MODEL, upstreams.openai, OPENAI_UPSTREAM_MODEL);
    await openaiModel(FAILING_MODEL, upstreams.failing, OPENAI_UPSTREAM_MODEL);
    await openaiModel(REFUSING_MODEL, upstreams.refusing, OPENAI_UPSTREAM_MODEL);
    await openaiModel(STREAM_MODEL, upstreams.stream, OPENAI_UPSTREAM_MODEL);
    const embedPk = await seed.createProviderKey({
      display_name: `${EMBED_MODEL}-pk`,
      provider: "openai",
      secret: "sk-mock",
      api_base: `${upstreams.embed.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: EMBED_MODEL,
      provider: "openai",
      model_name: EMBED_UPSTREAM_MODEL,
      provider_key_id: embedPk.id,
    });
    const anthropicPk = await seed.createProviderKey({
      display_name: `${ANTHROPIC_MODEL}-pk`,
      provider: "anthropic",
      adapter: "anthropic",
      secret: "sk-ant-mock",
      api_base: upstreams.anthropic.baseUrl,
    });
    await seed.createModel({
      display_name: ANTHROPIC_MODEL,
      provider: "anthropic",
      model_name: ANTHROPIC_UPSTREAM_MODEL,
      provider_key_id: anthropicPk.id,
    });
    // Seeded last: its authenticating implies every row above is loaded.
    await seed.createApiKey({ key_hash: CALLER_HASH, allowed_models: ["*"] });
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${CALLER}` },
      });
      await res.text();
      return res.status === 200;
    });
  }, 60_000);

  afterAll(async () => {
    await app?.exit();
    await otlp?.close();
    await dd?.close();
    await Promise.all(Object.values(upstreams).map((u) => u.close()));
  });

  async function post(path: string, body: unknown): Promise<{ status: number; requestId: string; text: string }> {
    const res = await fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: {
        ...(path === "/v1/messages" ? { "x-api-key": CALLER } : { authorization: `Bearer ${CALLER}` }),
        "content-type": "application/json",
        "user-agent": "semconv-e2e/1.0",
      },
      body: JSON.stringify(body),
    });
    const text = await res.text();
    const requestId = res.headers.get("x-aisix-request-id") ?? "";
    expect(requestId, `${path} returned a request id`).not.toBe("");
    return { status: res.status, requestId, text };
  }

  test("an OpenAI-shape chat: provider, upstream model, request parameters, inclusive usage, structured content", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const { status, requestId, text } = await post("/v1/chat/completions", {
      model: OPENAI_MODEL,
      messages: [
        { role: "system", content: "answer tersely" },
        { role: "user", content: "semconv please" },
      ],
      temperature: 0.3,
      top_p: 0.9,
      max_tokens: 64,
      stop: ["END"],
      seed: 7,
      frequency_penalty: 0.25,
      presence_penalty: 0.5,
      response_format: { type: "json_object" },
    });
    expect(status, text).toBe(200);
    const spans = await requestSpans(otlp!, requestId);
    const attempt = attemptSpan(spans);
    const a = attempt.attributes;

    expect(attempt.name).toBe(`chat ${OPENAI_UPSTREAM_MODEL}`);
    expect(a["gen_ai.operation.name"]).toBe("chat");
    expect(a["gen_ai.provider.name"]).toBe("openai");
    expect(a["gen_ai.request.model"]).toBe(OPENAI_UPSTREAM_MODEL);
    expect(a["aisix.requested_model"]).toBe(OPENAI_MODEL);
    expect(a["gen_ai.system"]).toBeUndefined();
    expect(a["server.address"]).toBe("127.0.0.1");
    expect(a["server.port"]).toBe(portOf(upstreams.openai.baseUrl));

    expect(a["gen_ai.request.temperature"]).toBe(0.3);
    expect(a["gen_ai.request.top_p"]).toBe(0.9);
    expect(a["gen_ai.request.max_tokens"]).toBe(64);
    expect(a["gen_ai.request.stop_sequences"]).toBe("END");
    expect(a["gen_ai.request.seed"]).toBe(7);
    expect(a["gen_ai.request.frequency_penalty"]).toBe(0.25);
    expect(a["gen_ai.request.presence_penalty"]).toBe(0.5);
    expect(a["gen_ai.output.type"]).toBe("json");
    expect(a["gen_ai.request.stream"]).toBeUndefined();

    // OpenAI shape: cached and written tokens are already inside prompt_tokens.
    expect(a["gen_ai.usage.input_tokens"]).toBe(120);
    expect(a["gen_ai.usage.cache_read.input_tokens"]).toBe(100);
    expect(a["gen_ai.usage.cache_creation.input_tokens"]).toBe(20);
    expect(a["gen_ai.usage.output_tokens"]).toBe(30);
    expect(a["gen_ai.usage.reasoning.output_tokens"]).toBe(12);
    expect(a["gen_ai.response.finish_reasons"]).toBe("stop");
    expect(a["gen_ai.response.model"]).toBe("gpt-4o-mini-2024-07-18");

    // System prompts that ride the chat history stay in the history.
    expect(JSON.parse(String(a["gen_ai.input.messages"]))).toEqual([
      { role: "system", parts: [{ type: "text", content: "answer tersely" }] },
      { role: "user", parts: [{ type: "text", content: "semconv please" }] },
    ]);
    expect(JSON.parse(String(a["gen_ai.output.messages"]))).toEqual([
      { role: "assistant", parts: [{ type: "text", content: "semconv reply" }], finish_reason: "stop" },
    ]);
    expect(a["gen_ai.prompt"]).toBeUndefined();
    expect(a["gen_ai.completion"]).toBeUndefined();

    for (const span of spans) {
      expect(span.statusCode, `${span.name} status`).toBe(STATUS_UNSET);
      expect(span.attributes["error.type"]).toBeUndefined();
      // The harness boots with `observability.service_name: aisix-e2e`.
      expect(span.resource["service.name"]).toBe("aisix-e2e");
      expect(span.resource["service.version"]).toBeTruthy();
      expect(span.scope.version).toBe(span.resource["service.version"]);
    }
  });

  test("the SERVER span follows the HTTP server conventions and only the attempt reads as a GenAI operation", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const { status, requestId } = await post("/v1/chat/completions", {
      model: OPENAI_MODEL,
      messages: [{ role: "user", content: "server span" }],
    });
    expect(status).toBe(200);
    const spans = await requestSpans(otlp!, requestId);
    const server = spans.find((s) => s.kind === KIND_SERVER)!;
    expect(server.name).toBe("POST /v1/chat/completions");
    expect(server.attributes["http.request.method"]).toBe("POST");
    expect(server.attributes["url.path"]).toBe("/v1/chat/completions");
    expect(server.attributes["url.scheme"]).toBe("http");
    expect(server.attributes["http.route"]).toBe("/v1/chat/completions");
    expect(server.attributes["http.response.status_code"]).toBe(200);
    expect(server.attributes["user_agent.original"]).toBe("semconv-e2e/1.0");
    expect(server.attributes["client.address"]).toBe("127.0.0.1");
    expect(server.attributes["aisix.client_source_ip"]).toBeUndefined();
    // No query string, no `url.query`.
    expect(server.attributes["url.query"]).toBeUndefined();

    const attempt = attemptSpan(spans);
    const genai = spans.filter((s) => s.attributes["gen_ai.operation.name"] !== undefined);
    expect(genai.map((s) => s.spanId)).toEqual([attempt.spanId]);
    for (const span of spans) {
      if (span.spanId === attempt.spanId) continue;
      expect(span.attributes["gen_ai.request.model"], span.name).toBeUndefined();
      expect(span.attributes["gen_ai.usage.input_tokens"], span.name).toBeUndefined();
    }
  });

  test("the SERVER span reports the query with credential values redacted", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const query =
      "key=AIza-live-secret&Signature=sig-live&sig=s2&X-Goog-Signature=gs&access_token=at&tenant=ops&Sig=kept";
    const { status, requestId, text } = await post(`/v1/chat/completions?${query}`, {
      model: OPENAI_MODEL,
      messages: [{ role: "user", content: "query please" }],
    });
    expect(status, text).toBe(200);
    const spans = await requestSpans(otlp!, requestId);
    const server = spans.find((s) => s.kind === KIND_SERVER)!;
    expect(server.attributes["url.query"]).toBe(
      "key=REDACTED&Signature=REDACTED&sig=REDACTED&X-Goog-Signature=REDACTED&access_token=REDACTED&tenant=ops&Sig=kept",
    );
    expect(server.attributes["url.path"]).toBe("/v1/chat/completions");
    for (const span of spans) {
      expect(JSON.stringify(span.attributes)).not.toContain("live-secret");
      expect(JSON.stringify(span.attributes)).not.toContain("sig-live");
    }
  });

  test("an Anthropic-shape usage folds cache read and creation into input tokens", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const { status, requestId, text } = await post("/v1/messages", {
      model: ANTHROPIC_MODEL,
      max_tokens: 128,
      top_k: 40,
      stop_sequences: ["STOP"],
      system: "be precise",
      tools: [{ name: "lookup", description: "look things up", input_schema: { type: "object" } }],
      messages: [{ role: "user", content: "fold my cache" }],
    });
    expect(status, text).toBe(200);
    const attempt = attemptSpan(await requestSpans(otlp!, requestId));
    const a = attempt.attributes;
    expect(attempt.name).toBe(`chat ${ANTHROPIC_UPSTREAM_MODEL}`);
    expect(a["gen_ai.provider.name"]).toBe("anthropic");
    expect(a["gen_ai.usage.input_tokens"]).toBe(350);
    expect(a["gen_ai.usage.cache_read.input_tokens"]).toBe(300);
    expect(a["gen_ai.usage.cache_creation.input_tokens"]).toBe(40);
    expect(a["gen_ai.usage.output_tokens"]).toBe(5);
    expect(a["gen_ai.request.max_tokens"]).toBe(128);
    expect(a["gen_ai.request.top_k"]).toBe(40);
    expect(a["gen_ai.request.stop_sequences"]).toBe("STOP");
    expect(JSON.parse(String(a["gen_ai.system_instructions"]))).toEqual([
      { type: "text", content: "be precise" },
    ]);
    expect(JSON.parse(String(a["gen_ai.tool.definitions"]))).toEqual([
      { type: "function", name: "lookup" },
    ]);
    expect(JSON.parse(String(a["gen_ai.input.messages"]))).toEqual([
      { role: "user", parts: [{ type: "text", content: "fold my cache" }] },
    ]);
  });

  test("the Datadog exporter's input tokens follow the same accounting", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const { status, requestId, text } = await post("/v1/messages", {
      model: ANTHROPIC_MODEL,
      max_tokens: 128,
      messages: [{ role: "user", content: "datadog fold" }],
    });
    expect(status, text).toBe(200);
    const log = await waitFor("the Datadog log of the Anthropic call", () =>
      dd!.logs.find((l) => l["aisix.request_id"] === requestId),
    );
    expect(log["gen_ai.usage.input_tokens"]).toBe(350);
    // The raw counters keep their own keys.
    expect(log["aisix.cache_read_tokens"]).toBe(300);
    expect(log["aisix.cache_creation_tokens"]).toBe(40);
  });

  test("an embeddings call is an embeddings span with its encoding and dimensions", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const { status, requestId, text } = await post("/v1/embeddings", {
      model: EMBED_MODEL,
      input: "embed me",
      encoding_format: "float",
      dimensions: 256,
    });
    expect(status, text).toBe(200);
    const attempt = attemptSpan(await requestSpans(otlp!, requestId));
    const a = attempt.attributes;
    expect(attempt.name).toBe(`embeddings ${EMBED_UPSTREAM_MODEL}`);
    expect(a["gen_ai.operation.name"]).toBe("embeddings");
    expect(a["gen_ai.provider.name"]).toBe("openai");
    expect(a["gen_ai.request.encoding_formats"]).toBe("float");
    expect(a["gen_ai.embeddings.dimension.count"]).toBe(256);
    expect(a["gen_ai.usage.input_tokens"]).toBe(4);
    expect(a["gen_ai.usage.output_tokens"]).toBeUndefined();
  });

  test("a streamed chat reports the stream flag and time to first chunk", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const { status, requestId } = await post("/v1/chat/completions", {
      model: STREAM_MODEL,
      stream: true,
      stream_options: { include_usage: true },
      messages: [{ role: "user", content: "stream please" }],
    });
    expect(status).toBe(200);
    const a = attemptSpan(await requestSpans(otlp!, requestId)).attributes;
    expect(a["gen_ai.request.stream"]).toBe(true);
    // The upstream held its first frame back 60ms.
    expect(Number(a["gen_ai.response.time_to_first_chunk"])).toBeGreaterThanOrEqual(0.05);
    expect(Number(a["gen_ai.response.time_to_first_chunk"])).toBeLessThan(30);
  });

  test("an upstream 5xx is an error on every span; an upstream 4xx is an error on the CLIENT spans only", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const failed = await post("/v1/chat/completions", {
      model: FAILING_MODEL,
      messages: [{ role: "user", content: "fail please" }],
    });
    expect(failed.status).toBeGreaterThanOrEqual(500);
    const failedSpans = await requestSpans(otlp!, failed.requestId);
    const failedAttempt = attemptSpan(failedSpans);
    expect(failedAttempt.statusCode).toBe(STATUS_ERROR);
    expect(failedAttempt.attributes["error.type"]).toBe("upstream_status");
    expect(failedAttempt.attributes["aisix.error_class"]).toBe("upstream_status");
    const failedServer = failedSpans.find((s) => s.kind === KIND_SERVER)!;
    expect(failedServer.statusCode).toBe(STATUS_ERROR);
    expect(failedServer.attributes["error.type"]).toBeTruthy();

    const refused = await post("/v1/chat/completions", {
      model: REFUSING_MODEL,
      messages: [{ role: "user", content: "refuse please" }],
    });
    expect(refused.status).toBe(400);
    const refusedSpans = await requestSpans(otlp!, refused.requestId);
    const refusedAttempt = attemptSpan(refusedSpans);
    expect(refusedAttempt.statusCode).toBe(STATUS_ERROR);
    expect(refusedAttempt.attributes["error.type"]).toBeTruthy();
    const refusedServer = refusedSpans.find((s) => s.kind === KIND_SERVER)!;
    expect(refusedServer.statusCode).toBe(STATUS_UNSET);
    expect(refusedServer.attributes["error.type"]).toBeUndefined();
    expect(refusedServer.attributes["http.response.status_code"]).toBe(400);
  });
});
