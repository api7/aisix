import { createHash } from "node:crypto";
import { createServer, type IncomingMessage, type Server } from "node:http";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  eventStreamException,
  slsLogsFor,
  spawnApp,
  startMockSls,
  waitConfigPropagation,
  waitForLogLine,
  waitForSlsLog,
  type MockSls,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: a provider names each call it answers with a request id of its own,
// in a response header — and that id is what its support asks for about a
// call that failed. A failed attempt has no response object to read an id
// from, and a token count's answer carries none, so the header is the only
// place it exists.
//
// The contract read back here, for every provider family the gateway
// dispatches to:
//   - each failed attempt's usage event carries the id of the response that
//     failed it (`provider_request_id`);
//   - a `/v1/messages/count_tokens` event carries the id of the count;
//   - the failed request's access-log line carries the id, and names the
//     provider the request was sent to rather than `unknown`, on every
//     route.

const CALLER = "sk-upstream-request-id-errors";
const CALLER_HASH = createHash("sha256").update(CALLER).digest("hex");
const CREDENTIAL_REF = "e2e";
const LOGSTORE = "upstream-request-id-store";

const OPENAI = "urid-openai";
const OPENAI_RETRIED = "urid-openai-retried";
const ANTHROPIC = "urid-anthropic";
const AZURE = "urid-azure";
const BEDROCK = "urid-bedrock-nova";
const BEDROCK_CLAUDE = "urid-bedrock-claude";
const VERTEX = "urid-vertex-gemini";
const DECISIONS = "urid-decisions";

// The id each upstream answers with, keyed off what it was asked, so a
// matching row proves WHICH response the id was read from.
let retriedCalls = 0;

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve) => {
    const chunks: Buffer[] = [];
    req.on("data", (c: Buffer) => chunks.push(c));
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
  });
}

async function startUpstream(): Promise<{ server: Server; baseUrl: string }> {
  const server = createServer(async (req, res) => {
    await readBody(req);
    const path = decodeURIComponent((req.url ?? "/").split("?")[0]);
    const fail = (status: number, headers: Record<string, string>, body: unknown) => {
      res.writeHead(status, { "content-type": "application/json", ...headers });
      res.end(JSON.stringify(body));
    };
    const openAiError = { error: { message: "rejected by the mock", type: "invalid_request_error" } };

    // Bedrock runtime: the AWS SDK's own operations.
    if (path.startsWith("/model/")) {
      const awsId = (tag: string) => ({ "x-amzn-requestid": `amzn-${tag}` });
      if (path.endsWith("/count-tokens")) {
        res.writeHead(200, { "content-type": "application/json", ...awsId("count") });
        res.end(JSON.stringify({ inputTokens: 21 }));
        return;
      }
      if (path.endsWith("/converse-stream")) {
        // Committed 200, then a modeled exception before any content.
        res.writeHead(200, { "content-type": "application/vnd.amazon.eventstream", ...awsId("stream") });
        res.end(eventStreamException("throttlingException", { message: "slow down" }));
        return;
      }
      const tag = path.endsWith("/converse") ? "converse" : "invoke";
      return fail(
        400,
        { ...awsId(tag), "x-amzn-errortype": "ValidationException:http://internal.amazon.com/coral/com.amazon.bedrock/" },
        { message: "The provided model identifier is invalid." },
      );
    }
    // Vertex AI.
    if (path.includes("/projects/")) {
      return fail(
        400,
        { "x-goog-request-id": "goog-generate" },
        { error: { code: 400, message: "Invalid argument.", status: "INVALID_ARGUMENT" } },
      );
    }
    // Azure OpenAI sends two ids; `x-request-id` is the one asked for first.
    if (path.includes("/openai/deployments/")) {
      return fail(429, { "x-request-id": "azure-chat", "apim-request-id": "azure-apim" }, {
        error: { code: "429", message: "Rate limit is exceeded." },
      });
    }
    // Anthropic.
    if (path.startsWith("/ant/")) {
      if (path.endsWith("/count_tokens")) {
        res.writeHead(200, { "content-type": "application/json", "request-id": "req_ant_count" });
        res.end(JSON.stringify({ input_tokens: 13 }));
        return;
      }
      return fail(400, { "request-id": "req_ant_messages" }, {
        type: "error",
        error: { type: "invalid_request_error", message: "rejected by the mock" },
      });
    }
    // Decisions, on its own provider.
    if (path.startsWith("/dec/")) {
      return fail(400, { "x-request-id": "dec-systemone" }, { detail: "rejected by the mock" });
    }
    // OpenAI, retried: every attempt answers with an id of its own.
    if (path.startsWith("/oai-retried/")) {
      retriedCalls += 1;
      return fail(503, { "x-request-id": `oai-retried-${retriedCalls}` }, openAiError);
    }
    // OpenAI: the id names the route it answered.
    if (path.startsWith("/oai/")) {
      return fail(400, { "x-request-id": `oai${path.slice("/oai/v1".length).replaceAll("/", "-")}` }, openAiError);
    }
    fail(404, {}, { message: `no route ${path}` });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const addr = server.address();
  if (addr === null || typeof addr === "string") throw new Error("no port");
  return { server, baseUrl: `http://127.0.0.1:${addr.port}` };
}

type Sent = { status: number; requestId: string; text: string };

describe("the provider's request id on failed attempts and token counts", () => {
  let app: SpawnedApp | undefined;
  let upstream: Server | undefined;
  let sls: MockSls | undefined;
  let etcdReachable = false;
  const auth = { authorization: `Bearer ${CALLER}` };

  async function json(path: string, body: Record<string, unknown>): Promise<Sent> {
    const res = await fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: { ...auth, "content-type": "application/json", "anthropic-version": "2023-06-01" },
      body: JSON.stringify(body),
    });
    const text = await res.text();
    return { status: res.status, requestId: res.headers.get("x-aisix-request-id") ?? "", text };
  }

  async function multipart(path: string, fields: Record<string, string>, file: string): Promise<Sent> {
    const form = new FormData();
    for (const [k, v] of Object.entries(fields)) form.set(k, v);
    form.set(file, new Blob([new Uint8Array([0x49, 0x44, 0x33])], { type: "application/octet-stream" }), "f.bin");
    const res = await fetch(`${app!.proxyUrl}${path}`, { method: "POST", headers: auth, body: form });
    const text = await res.text();
    return { status: res.status, requestId: res.headers.get("x-aisix-request-id") ?? "", text };
  }

  function rows(requestId: string): Array<Map<string, string>> {
    return slsLogsFor(sls!, LOGSTORE).filter((l) => l.get("request_id") === requestId);
  }

  async function rowsFor(requestId: string, count: number): Promise<Array<Map<string, string>>> {
    expect(requestId, "x-aisix-request-id").not.toBe("");
    await waitForSlsLog(
      sls!,
      LOGSTORE,
      () => rows(requestId).length >= count,
      `${count} usage row(s) for ${requestId}`,
      15_000,
    );
    return rows(requestId).sort((a, b) => Number(a.get("attempt_index") ?? 0) - Number(b.get("attempt_index") ?? 0));
  }

  async function accessLine(requestId: string): Promise<string> {
    return waitForLogLine(
      app!,
      (l) => l.includes("proxy request completed") && field(l, "request_id") === requestId,
      `the access-log line of ${requestId}`,
    );
  }

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    const up = await startUpstream();
    upstream = up.server;
    sls = await startMockSls();
    app = await spawnApp({
      extraEnv: {
        RUST_LOG: "info",
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: "mock-ak-id",
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: "mock-ak-secret",
      },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createObservabilityExporter({
      name: "urid-sls",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: "urid-proj",
      logstore: LOGSTORE,
      credential_ref: CREDENTIAL_REF,
    });

    const pk = (display_name: string, extra: Record<string, unknown>) =>
      seed.createProviderKey({ display_name, secret: "sk-mock", ...extra });
    const openai = await pk("urid-openai-pk", { api_base: `${up.baseUrl}/oai/v1` });
    const retried = await pk("urid-openai-retried-pk", { api_base: `${up.baseUrl}/oai-retried/v1` });
    const anthropic = await pk("urid-anthropic-pk", {
      provider: "anthropic",
      adapter: "anthropic",
      api_base: `${up.baseUrl}/ant`,
    });
    const azure = await pk("urid-azure-pk", { provider: "azure-openai", adapter: "azure-openai", api_base: up.baseUrl });
    const bedrock = await pk("urid-bedrock-pk", {
      provider: "bedrock",
      adapter: "bedrock",
      secret: JSON.stringify({ access_key_id: "AKIA-urid", secret_access_key: "sk-urid", region: "us-west-2" }),
      api_base: up.baseUrl,
    });
    const vertex = await pk("urid-vertex-pk", {
      provider: "google",
      adapter: "vertex",
      secret: JSON.stringify({ access_token: "ya29.urid", project: "proj-e2e", region: "us-central1" }),
      api_base: up.baseUrl,
    });
    const decisions = await pk("urid-decisions-pk", {
      provider: "typesafe",
      adapter: undefined,
      api_base: `${up.baseUrl}/dec`,
    });
    for (const [display_name, key, provider, model_name, retries] of [
      [OPENAI, openai, "openai", "gpt-4o-mini", 0],
      [OPENAI_RETRIED, retried, "openai", "gpt-4o-mini", 1],
      [ANTHROPIC, anthropic, "anthropic", "claude-sonnet-4-5", 0],
      [AZURE, azure, "azure-openai", "gpt-4o-mini", 0],
      [BEDROCK, bedrock, "bedrock", "amazon.nova-lite-v1:0", 0],
      [BEDROCK_CLAUDE, bedrock, "bedrock", "global.anthropic.claude-sonnet-4-5-20250929-v1:0", 0],
      [VERTEX, vertex, "google", "gemini-2.5-flash", 0],
      [DECISIONS, decisions, "typesafe", "jev-latest", 0],
    ] as const) {
      await seed.createModel({ display_name, provider, model_name, provider_key_id: key.id, retries });
    }

    // Seeded last: this key authenticating implies the whole seed set.
    await seed.createApiKey({ key_hash: CALLER_HASH, allowed_models: ["*"] });
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, { headers: auth });
      await res.text();
      return res.status === 200;
    });
  });

  afterAll(async () => {
    await app?.exit();
    await new Promise<void>((resolve) => (upstream ? upstream.close(() => resolve()) : resolve()));
    await sls?.close();
  });

  const chat = (model: string, stream = false) => ({ model, stream, messages: [{ role: "user", content: "hi" }] });
  const messages = (model: string) => ({ model, max_tokens: 16, messages: [{ role: "user", content: "hi" }] });

  // [case, send, provider the line names, the id the upstream answered with].
  const families: Array<[string, () => Promise<Sent>, string, string]> = [
    ["OpenAI, /v1/chat/completions", () => json("/v1/chat/completions", chat(OPENAI)), "openai", "oai-chat-completions"],
    ["Anthropic, /v1/messages", () => json("/v1/messages", messages(ANTHROPIC)), "anthropic", "req_ant_messages"],
    ["Azure OpenAI, /v1/chat/completions", () => json("/v1/chat/completions", chat(AZURE)), "azure-openai", "azure-chat"],
    ["Bedrock Converse, /v1/chat/completions", () => json("/v1/chat/completions", chat(BEDROCK)), "bedrock", "amzn-converse"],
    [
      "Bedrock ConverseStream, an exception before the first token",
      () => json("/v1/chat/completions", chat(BEDROCK, true)),
      "bedrock",
      "amzn-stream",
    ],
    ["Claude on Bedrock, /v1/messages", () => json("/v1/messages", messages(BEDROCK_CLAUDE)), "bedrock", "amzn-invoke"],
    ["Vertex AI Gemini, /v1/chat/completions", () => json("/v1/chat/completions", chat(VERTEX)), "google", "goog-generate"],
  ];

  for (const [name, send, provider, upstreamId] of families) {
    test(`${name}: the failed attempt carries the provider's request id`, async (ctx) => {
      if (!etcdReachable || !app) return ctx.skip();
      const res = await send();
      const [row] = await rowsFor(res.requestId, 1);
      const shown = JSON.stringify(Object.fromEntries(row));
      // A stream the upstream failed before its first token has already
      // answered `200`; the row names the failure either way.
      expect(row.get("error_class"), shown).toBeTruthy();
      expect(row.get("provider_request_id"), shown).toBe(upstreamId);

      const line = await accessLine(res.requestId);
      expect(field(line, "provider"), line).toBe(provider);
      expect(field(line, "provider_request_id"), line).toBe(upstreamId);
    });
  }

  test("each retried attempt carries the id of its own response", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const before = retriedCalls;
    const res = await json("/v1/chat/completions", chat(OPENAI_RETRIED));
    expect(res.status, res.text).toBeGreaterThanOrEqual(500);
    expect(retriedCalls - before, "one retry after the first failure").toBe(2);
    const attempts = await rowsFor(res.requestId, 2);
    expect(attempts.map((r) => r.get("provider_request_id"))).toEqual([
      `oai-retried-${before + 1}`,
      `oai-retried-${before + 2}`,
    ]);
    // The line names the attempt whose failure the caller was served.
    const line = await accessLine(res.requestId);
    expect(field(line, "provider_request_id"), line).toBe(`oai-retried-${before + 2}`);
  });

  for (const [name, model, upstreamId] of [
    ["Anthropic", ANTHROPIC, "req_ant_count"],
    ["Claude on Bedrock", BEDROCK_CLAUDE, "amzn-count"],
  ] as const) {
    test(`count_tokens on ${name} records the provider's request id`, async (ctx) => {
      if (!etcdReachable || !app) return ctx.skip();
      const res = await json("/v1/messages/count_tokens", { model, messages: [{ role: "user", content: "hi" }] });
      expect(res.status, res.text).toBe(200);
      const [row] = await rowsFor(res.requestId, 1);
      expect(row.get("provider_request_id"), JSON.stringify(Object.fromEntries(row))).toBe(upstreamId);
      const line = await accessLine(res.requestId);
      expect(field(line, "provider_request_id"), line).toBe(upstreamId);
    });
  }

  // Every route's failure branch writes its own line, so each is driven.
  // `/v1/realtime` is absent: it is a WebSocket upgrade this mock does not
  // serve.
  const routes: Array<[string, () => Promise<Sent>, string, string]> = [
    ["/v1/chat/completions", () => json("/v1/chat/completions", chat(OPENAI)), "openai", "oai-chat-completions"],
    [
      "/v1/chat/completions, streamed",
      () => json("/v1/chat/completions", chat(OPENAI, true)),
      "openai",
      "oai-chat-completions",
    ],
    ["/v1/completions", () => json("/v1/completions", { model: OPENAI, prompt: "hi" }), "openai", "oai-completions"],
    ["/v1/embeddings", () => json("/v1/embeddings", { model: OPENAI, input: "hi" }), "openai", "oai-embeddings"],
    [
      "/v1/rerank",
      () => json("/v1/rerank", { model: OPENAI, query: "q", documents: ["a", "b"] }),
      "openai",
      "oai-rerank",
    ],
    [
      "/v1/images/generations",
      () => json("/v1/images/generations", { model: OPENAI, prompt: "a cat" }),
      "openai",
      "oai-images-generations",
    ],
    [
      "/v1/images/edits",
      () => multipart("/v1/images/edits", { model: OPENAI, prompt: "a hat" }, "image"),
      "openai",
      "oai-images-edits",
    ],
    [
      "/v1/audio/speech",
      () => json("/v1/audio/speech", { model: OPENAI, input: "hello", voice: "alloy" }),
      "openai",
      "oai-audio-speech",
    ],
    [
      "/v1/audio/transcriptions",
      () => multipart("/v1/audio/transcriptions", { model: OPENAI }, "file"),
      "openai",
      "oai-audio-transcriptions",
    ],
    [
      "/v1/audio/translations",
      () => multipart("/v1/audio/translations", { model: OPENAI }, "file"),
      "openai",
      "oai-audio-translations",
    ],
    ["/v1/responses", () => json("/v1/responses", { model: OPENAI, input: "hi" }), "openai", "oai-responses"],
    [
      "/v1/responses, streamed",
      () => json("/v1/responses", { model: OPENAI, input: "hi", stream: true }),
      "openai",
      "oai-responses",
    ],
    ["/v1/messages", () => json("/v1/messages", messages(OPENAI)), "openai", "oai-chat-completions"],
    ["/v1/videos", () => json("/v1/videos", { model: OPENAI, prompt: "a boat" }), "openai", "oai-videos"],
    ["/v1/systemone", () => json("/v1/systemone", systemOneBody(DECISIONS)), "typesafe", "dec-systemone"],
  ];

  for (const [route, send, provider, upstreamId] of routes) {
    test(`${route}: a failed request's access-log line names the provider and its request id`, async (ctx) => {
      if (!etcdReachable || !app) return ctx.skip();
      const res = await send();
      expect(res.status, `${route} did not reach the failing upstream: ${res.text}`).toBeGreaterThanOrEqual(400);
      const line = await accessLine(res.requestId);
      expect(field(line, "provider"), line).toBe(provider);
      expect(field(line, "provider_request_id"), line).toBe(upstreamId);
    });
  }

  test("a request that never reached an upstream carries neither", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const res = await json("/v1/chat/completions", chat("urid-no-such-model"));
    expect(res.status).toBe(404);
    const line = await accessLine(res.requestId);
    expect(field(line, "provider"), line).toBeUndefined();
    expect(field(line, "provider_request_id"), line).toBeUndefined();
  });
});

function systemOneBody(model: string) {
  return {
    model,
    state: "The API returns 500 on every call since this morning.",
    questions: {
      is_bug: {
        type: "noul",
        instructions: "Is this a software bug report?",
        criteria: { true: "the software misbehaves", false: null },
      },
    },
  };
}

function field(line: string, name: string): string | undefined {
  const m = line.match(new RegExp(`\\b${name}=(?:"([^"]*)"|([^\\s]+))`));
  if (!m) return undefined;
  return m[1] ?? m[2];
}
