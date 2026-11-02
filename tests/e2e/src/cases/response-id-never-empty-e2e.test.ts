import { createHash } from "node:crypto";
import { createServer, type Server } from "node:http";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  eventStream,
  spawnApp,
  startMockSls,
  waitConfigPropagation,
  waitForSlsLog,
  type MockSls,
  type SpawnedApp,
} from "../harness/index.js";

// A chat, messages or responses answer always carries an id. A Gemini answer's
// is its `responseId`; an upstream that reports none (Gemini without the field,
// Bedrock Converse) gets one from the gateway, which is not mistaken for the
// provider's in the usage event. On a chat stream the upstream's first id holds
// from the frame that reports it on, and empty frames ahead of it share one
// generated id.

const CALLER = "sk-response-id-never-empty";
const CALLER_HASH = createHash("sha256").update(CALLER).digest("hex");
const CREDENTIAL_REF = "mock";
const LOGSTORE = "response-id";
const RESPONSE_ID = "h-Pq_2xY7-Lm0Nb4Vc9Zt1A";

const USAGE = { promptTokenCount: 3, candidatesTokenCount: 2, totalTokenCount: 5 };
const frame = (text: string, responseId: string | undefined, finishReason?: string) => ({
  ...(responseId === undefined ? {} : { responseId }),
  candidates: [{ content: { role: "model", parts: [{ text }] }, ...(finishReason ? { finishReason } : {}) }],
  ...(finishReason ? { usageMetadata: USAGE } : {}),
});

/** Which `responseId`s the Vertex mock stamps on a stream's three frames. */
const GEMINI_MODELS: Record<string, Array<string | undefined>> = {
  "gemini-with-id": [RESPONSE_ID, RESPONSE_ID, RESPONSE_ID],
  "gemini-without-id": [undefined, undefined, undefined],
  "gemini-late-id": [undefined, RESPONSE_ID, RESPONSE_ID],
};

/** An OpenAI-shape stream whose frames report these ids, in order. */
const OPENAI_FRAME_IDS = ["", "", "up-first", "", "up-other"];
const OPENAI_STREAM = OPENAI_FRAME_IDS.map((id, i) =>
  JSON.stringify({
    id,
    object: "chat.completion.chunk",
    created: 1_700_000_000,
    model: "gpt-4o-mini",
    choices: [
      {
        index: 0,
        delta: i === 0 ? { role: "assistant", content: "o" } : { content: "k" },
        finish_reason: i === OPENAI_FRAME_IDS.length - 1 ? "stop" : null,
      },
    ],
  }),
);

const CONVERSE_REPLY = {
  output: { message: { role: "assistant", content: [{ text: "ok" }] } },
  stopReason: "end_turn",
  usage: { inputTokens: 3, outputTokens: 2, totalTokens: 5 },
  metrics: { latencyMs: 5 },
};
const CONVERSE_STREAM = eventStream([
  ["messageStart", { role: "assistant" }],
  ["contentBlockDelta", { contentBlockIndex: 0, delta: { text: "ok" } }],
  ["contentBlockStop", { contentBlockIndex: 0 }],
  ["messageStop", { stopReason: "end_turn" }],
  ["metadata", { usage: { inputTokens: 3, outputTokens: 2, totalTokens: 5 }, metrics: { latencyMs: 5 } }],
]);

function sseData(raw: string): any[] {
  return raw
    .split(/\r?\n/)
    .filter((l) => l.startsWith("data: ") && !l.includes("[DONE]"))
    .map((l) => JSON.parse(l.slice(6)));
}

describe("response ids", () => {
  let app: SpawnedApp | undefined;
  let upstream: Server | undefined;
  let sls: MockSls | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = createServer((req, res) => {
      req.resume();
      req.on("end", () => {
        const path = (req.url ?? "").split("?")[0];
        if (path.endsWith("/chat/completions")) {
          res.writeHead(200, { "content-type": "text/event-stream" });
          res.end(`${OPENAI_STREAM.map((f) => `data: ${f}\n\n`).join("")}data: [DONE]\n\n`);
          return;
        }
        if (path.includes("/model/")) {
          if (path.endsWith("/converse-stream")) {
            res.writeHead(200, { "content-type": "application/vnd.amazon.eventstream" });
            res.end(CONVERSE_STREAM);
          } else {
            res.writeHead(200, { "content-type": "application/json" });
            res.end(JSON.stringify(CONVERSE_REPLY));
          }
          return;
        }
        const ids = Object.entries(GEMINI_MODELS).find(([name]) => path.includes(`/${name}:`))![1];
        if (path.endsWith(":streamGenerateContent")) {
          const frames = [frame("o", ids[0]), frame("k", ids[1]), frame("", ids[2], "STOP")];
          res.writeHead(200, { "content-type": "text/event-stream" });
          res.end(frames.map((f) => `data: ${JSON.stringify(f)}\r\n\r\n`).join(""));
        } else {
          res.writeHead(200, { "content-type": "application/json" });
          res.end(JSON.stringify(frame("ok", ids[2], "STOP")));
        }
      });
    });
    await new Promise<void>((resolve) => upstream!.listen(0, "127.0.0.1", resolve));
    const addr = upstream.address();
    if (addr === null || typeof addr === "string") throw new Error("no port");
    const base = `http://127.0.0.1:${addr.port}`;

    sls = await startMockSls();
    app = await spawnApp({
      extraEnv: {
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: "LTAI_mock_ak",
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: "mock_ak_secret",
      },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createObservabilityExporter({
      name: "id-sls",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: "aisix-e2e-obs",
      logstore: LOGSTORE,
      credential_ref: CREDENTIAL_REF,
    });
    const vertex = await seed.createProviderKey({
      display_name: "id-vertex-pk",
      provider: "google",
      adapter: "vertex",
      secret: JSON.stringify({ access_token: "ya29.response-id-e2e", project: "proj-e2e", region: "us-central1" }),
      api_base: base,
    });
    for (const name of Object.keys(GEMINI_MODELS)) {
      await seed.createModel({ display_name: name, provider: "google", model_name: name, provider_key_id: vertex.id });
    }
    const bedrock = await seed.createProviderKey({
      display_name: "id-bedrock-pk",
      provider: "bedrock",
      adapter: "bedrock",
      secret: JSON.stringify({ access_key_id: "AKIA-response-id", secret_access_key: "sk-response-id", region: "us-east-1" }),
      api_base: base,
    });
    await seed.createModel({
      display_name: "converse-nova",
      provider: "bedrock",
      model_name: "us.amazon.nova-lite-v1:0",
      provider_key_id: bedrock.id,
    });
    const openai = await seed.createProviderKey({
      display_name: "id-openai-pk",
      secret: "sk-mock",
      api_base: `${base}/v1`,
    });
    await seed.createModel({
      display_name: "openai-late-id",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: openai.id,
    });
    // Seeded last: this key authenticating implies the whole seed set landed.
    await seed.createApiKey({ key_hash: CALLER_HASH, allowed_models: ["*"] });
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, { headers: { authorization: `Bearer ${CALLER}` } });
      await res.text();
      return res.status === 200;
    });
  }, 60_000);

  afterAll(async () => {
    await app?.exit();
    await sls?.close();
    await new Promise<void>((resolve) => (upstream ? upstream.close(() => resolve()) : resolve()));
  });

  async function post(path: string, body: Record<string, unknown>): Promise<{ raw: string; requestId: string }> {
    const res = await fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: { authorization: `Bearer ${CALLER}`, "x-api-key": CALLER, "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    const raw = await res.text();
    expect(res.status, raw.slice(0, 500)).toBe(200);
    return { raw, requestId: res.headers.get("x-aisix-request-id") ?? "" };
  }

  const chat = (model: string, stream = false) =>
    post("/v1/chat/completions", { model, stream, messages: [{ role: "user", content: "hi" }] });
  const messages = (model: string, stream = false) =>
    post("/v1/messages", { model, stream, max_tokens: 16, messages: [{ role: "user", content: "hi" }] });
  const responses = (model: string, stream = false) => post("/v1/responses", { model, stream, input: "hi" });

  async function providerRequestId(requestId: string): Promise<string | undefined> {
    const row = await waitForSlsLog(sls!, LOGSTORE, (l) => l.get("request_id") === requestId, "usage row");
    return row.get("provider_request_id");
  }

  /** The one id every frame of a chat stream carries. */
  function chatStreamId(raw: string): string {
    const frames = sseData(raw);
    expect(frames.length).toBeGreaterThan(1);
    const ids = new Set(frames.map((f) => f.id));
    expect([...ids], "every frame carries one id").toHaveLength(1);
    return [...ids][0];
  }

  describe("Gemini with responseId", () => {
    test("chat answers with it", async (ctx) => {
      if (!etcdReachable || !app) return ctx.skip();
      const { raw, requestId } = await chat("gemini-with-id");
      expect(JSON.parse(raw).id).toBe(RESPONSE_ID);
      expect(await providerRequestId(requestId)).toBe(RESPONSE_ID);
    });

    test("every chat stream frame carries it", async (ctx) => {
      if (!etcdReachable || !app) return ctx.skip();
      const { raw, requestId } = await chat("gemini-with-id", true);
      expect(chatStreamId(raw)).toBe(RESPONSE_ID);
      expect(await providerRequestId(requestId)).toBe(RESPONSE_ID);
    });

    test("messages answers with it", async (ctx) => {
      if (!etcdReachable || !app) return ctx.skip();
      expect(JSON.parse((await messages("gemini-with-id")).raw).id).toBe(RESPONSE_ID);
    });
  });

  describe("an upstream that reports no id", () => {
    describe.each(["gemini-without-id", "converse-nova"])("%s", (model) => {
      test("chat gets a generated id", async (ctx) => {
        if (!etcdReachable || !app) return ctx.skip();
        const { raw, requestId } = await chat(model);
        expect(JSON.parse(raw).id).toMatch(/^chatcmpl-[0-9a-f]{32}$/);
        expect(await providerRequestId(requestId), "a generated id is not the provider's").toBeFalsy();
      });

      test("a chat stream keeps one generated id", async (ctx) => {
        if (!etcdReachable || !app) return ctx.skip();
        const { raw, requestId } = await chat(model, true);
        expect(chatStreamId(raw)).toMatch(/^chatcmpl-[0-9a-f]{32}$/);
        expect(await providerRequestId(requestId), "a generated id is not the provider's").toBeFalsy();
      });

      test("messages gets a generated id", async (ctx) => {
        if (!etcdReachable || !app) return ctx.skip();
        expect(JSON.parse((await messages(model)).raw).id).toMatch(/^msg_[0-9a-f]{32}$/);
      });

      test("two answers do not share an id", async (ctx) => {
        if (!etcdReachable || !app) return ctx.skip();
        const [a, b] = await Promise.all([chat(model), chat(model)]);
        expect(JSON.parse(a.raw).id).not.toBe(JSON.parse(b.raw).id);
      });
    });
  });

  test("Gemini: frames ahead of a late responseId share a generated id, then it holds", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const { raw, requestId } = await chat("gemini-late-id", true);
    const ids = sseData(raw).map((f) => f.id);
    const leading = ids.filter((id) => id !== RESPONSE_ID);
    expect(new Set(leading).size, "one generated id ahead of the upstream's").toBe(1);
    expect(leading[0]).toMatch(/^chatcmpl-[0-9a-f]{32}$/);
    expect(ids.slice(0, leading.length)).toEqual(leading);
    expect(ids.slice(leading.length).length, "frames carry the upstream's id").toBeGreaterThan(0);
    expect(await providerRequestId(requestId), "the upstream's id is recorded").toBe(RESPONSE_ID);
  });

  test("the upstream's first id holds whatever later frames report", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const ids = sseData((await chat("openai-late-id", true)).raw).map((f) => f.id);
    // One frame per upstream frame: two empty, then "up-first", "", "up-other".
    expect(ids).toHaveLength(OPENAI_FRAME_IDS.length);
    expect(ids[0]).toMatch(/^chatcmpl-[0-9a-f]{32}$/);
    expect(ids[1]).toBe(ids[0]);
    expect(ids.slice(2)).toEqual(["up-first", "up-first", "up-first"]);
  });

  describe.each(["gemini-with-id", "gemini-without-id", "converse-nova"])("%s on the other streams", (model) => {
    test("a messages stream starts with an id", async (ctx) => {
      if (!etcdReachable || !app) return ctx.skip();
      const start = sseData((await messages(model, true)).raw).find((f) => f.type === "message_start");
      expect(start.message.id).toMatch(/^msg_[0-9a-f]{32}$/);
    });

    test("responses carries an id, streamed or not", async (ctx) => {
      if (!etcdReachable || !app) return ctx.skip();
      expect(JSON.parse((await responses(model)).raw).id).toMatch(/^resp_/);
      const completed = sseData((await responses(model, true)).raw).find((f) => f.type === "response.completed");
      expect(completed.response.id).toMatch(/^resp_/);
    });
  });
});
