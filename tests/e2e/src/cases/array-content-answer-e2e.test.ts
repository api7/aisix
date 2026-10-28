import { createHash } from "node:crypto";
import { createServer, type Server } from "node:http";
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

// An OpenAI-compatible upstream may answer a non-streamed chat completion
// with `message.content` as an array of parts rather than a string (Mistral,
// Databricks). Whatever protocol the caller speaks, the answer's text must
// reach it, and must reach the content-capturing exporters too.

const CALLER_PLAINTEXT = "sk-array-content-answer";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");
const HEADERS = {
  authorization: `Bearer ${CALLER_PLAINTEXT}`,
  "content-type": "application/json",
};

const ANSWER = "Paris is the capital of France.";
// Long enough that the capture cap below cuts it, so the exporter reports
// the answer as flat text instead of as a structured message.
const LONG_ANSWER = "array-answer-".repeat(30);
const CONTENT_MAX_BYTES = 128;

const arrayReply = (text: string) => ({
  id: "chatcmpl-array",
  object: "chat.completion",
  created: 1_700_000_000,
  model: "magistral-medium",
  choices: [
    {
      index: 0,
      // Split across two text parts: the answer is their concatenation.
      message: {
        role: "assistant",
        content: [
          { type: "text", text: text.slice(0, 10) },
          { type: "text", text: text.slice(10) },
        ],
      },
      finish_reason: "stop",
    },
  ],
  usage: { prompt_tokens: 7, completion_tokens: 8, total_tokens: 15 },
});

// Reasoning answered as content parts, in the two documented shapes:
// Mistral's thinking chunk (a list of text chunks) and Databricks' reasoning
// block (a list of summary_text entries).
const THINKING = "The user asks for the capital of France, which is Paris.";
const REASONING_PARTS = {
  mistral: { type: "thinking", thinking: [{ type: "text", text: THINKING }] },
  databricks: { type: "reasoning", summary: [{ type: "summary_text", text: THINKING, signature: "sig" }] },
} as const;

const reasoningReply = (part: unknown, withUsage = true) => ({
  id: "chatcmpl-reasoning",
  object: "chat.completion",
  created: 1_700_000_000,
  model: "magistral-medium",
  choices: [
    {
      index: 0,
      message: { role: "assistant", content: [part, { type: "text", text: ANSWER }] },
      finish_reason: "stop",
    },
  ],
  ...(withUsage ? { usage: { prompt_tokens: 7, completion_tokens: 30, total_tokens: 37 } } : {}),
});

// Long enough that its tokens dominate the estimate of an answer that
// reports no usage.
const LONG_THINKING = "Considering every European capital in turn before settling on one. ".repeat(20);

interface OtlpReceiver {
  url: string;
  spanAttrs: Array<Record<string, string>>;
  errors: string[];
  close(): Promise<void>;
}

async function startOtlpReceiver(): Promise<OtlpReceiver> {
  const spanAttrs: Array<Record<string, string>> = [];
  const errors: string[] = [];
  const server: Server = createServer((req, res) => {
    let raw = "";
    req.on("data", (c: Buffer) => (raw += c.toString("utf8")));
    req.on("end", () => {
      try {
        const body = JSON.parse(raw);
        for (const rs of body.resourceSpans ?? []) {
          for (const ss of rs.scopeSpans ?? []) {
            for (const span of ss.spans ?? []) {
              const attrs: Record<string, string> = {};
              for (const a of span.attributes ?? []) {
                const v = a.value ?? {};
                attrs[a.key] = v.arrayValue
                  ? JSON.stringify((v.arrayValue.values ?? []).map((e: any) => e.stringValue))
                  : (v.stringValue ?? String(v.intValue ?? v.boolValue ?? ""));
              }
              spanAttrs.push(attrs);
            }
          }
        }
      } catch (err) {
        errors.push(String(err));
        res.statusCode = 400;
        res.end("{}");
        return;
      }
      res.statusCode = 200;
      res.end("{}");
    });
  });
  const port = await pickFreePort();
  await new Promise<void>((resolve) => server.listen(port, "127.0.0.1", resolve));
  return {
    url: `http://127.0.0.1:${port}/v1/traces`,
    spanAttrs,
    errors,
    close: () => new Promise<void>((resolve, reject) => server.close((err) => (err ? reject(err) : resolve()))),
  };
}

async function waitForSpans(
  recv: OtlpReceiver,
  predicate: (attrs: Record<string, string>) => boolean,
  count: number,
  timeoutMs = 10_000,
): Promise<Array<Record<string, string>>> {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    const hits = recv.spanAttrs.filter(predicate);
    if (hits.length >= count) return hits;
    if (Date.now() > deadline) {
      throw new Error(
        `${hits.length}/${count} matching spans within timeout; ${recv.spanAttrs.length} spans seen; ` +
          `malformed payloads: ${JSON.stringify(recv.errors)}`,
      );
    }
    await new Promise((r) => setTimeout(r, 50));
  }
}

const promptOf = (a: Record<string, string>) => a["gen_ai.input.messages"] ?? a["aisix.content.prompt"] ?? "";

async function post(app: SpawnedApp, path: string, body: unknown) {
  return fetch(`${app.proxyUrl}${path}`, { method: "POST", headers: HEADERS, body: JSON.stringify(body) });
}

describe("an answer whose content is an array of parts", () => {
  let app: SpawnedApp | undefined;
  let otlp: OtlpReceiver | undefined;
  let otlpWide: OtlpReceiver | undefined;
  let etcdReachable = false;
  const up: Record<string, OpenAiUpstream> = {};

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    up.short = await startOpenAiUpstream({ nonStreamBody: arrayReply(ANSWER) });
    up.long = await startOpenAiUpstream({ nonStreamBody: arrayReply(LONG_ANSWER) });
    up.cached = await startOpenAiUpstream({ nonStreamBody: arrayReply(LONG_ANSWER) });
    up.mistral = await startOpenAiUpstream({ nonStreamBody: reasoningReply(REASONING_PARTS.mistral) });
    up.databricks = await startOpenAiUpstream({ nonStreamBody: reasoningReply(REASONING_PARTS.databricks) });
    up.mistralCached = await startOpenAiUpstream({ nonStreamBody: reasoningReply(REASONING_PARTS.mistral) });
    up.noUsage = await startOpenAiUpstream({
      nonStreamBody: reasoningReply({ type: "thinking", thinking: [{ type: "text", text: LONG_THINKING }] }, false),
    });
    otlp = await startOtlpReceiver();
    otlpWide = await startOtlpReceiver();

    app = await spawnApp({ exposeDebugHeaders: true });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    const model = async (name: string, upstream: OpenAiUpstream) => {
      const pk = await seed.createProviderKey({
        display_name: `${name}-pk`,
        secret: "sk-mock",
        api_base: `${upstream.baseUrl}/v1`,
      });
      await seed.createModel({
        display_name: name,
        // Not `openai`: `/v1/responses` reaches it through the chat bridge.
        provider: "mistral",
        model_name: "magistral-medium",
        provider_key_id: pk.id,
      });
    };
    await model("ac-short", up.short);
    await model("ac-long", up.long);
    await model("ac-cached", up.cached);
    await model("ac-mistral", up.mistral);
    await model("ac-databricks", up.databricks);
    await model("ac-no-usage", up.noUsage);
    await model("ac-mistral-cached", up.mistralCached);
    await seed.createCachePolicy({ name: "ac-cache", enabled: true, applies_to: "model:ac-cached" });
    await seed.createCachePolicy({
      name: "ac-cache-reasoning",
      enabled: true,
      applies_to: "model:ac-mistral-cached",
    });
    await seed.createObservabilityExporter({
      name: "ac-otlp",
      enabled: true,
      kind: "otlp_http",
      endpoint: otlp.url,
      content_mode: "full",
      content_max_bytes: CONTENT_MAX_BYTES,
    });
    // A cap the structured answer fits under, so it is reported as messages.
    await seed.createObservabilityExporter({
      name: "ac-otlp-wide",
      enabled: true,
      kind: "otlp_http",
      endpoint: otlpWide.url,
      content_mode: "full",
      content_max_bytes: 8192,
    });
    // Seeded last: the key authenticating implies the whole seed set is live.
    await seed.createApiKey({ key_hash: CALLER_KEY_HASH, allowed_models: ["*"] });
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, { headers: HEADERS });
      await res.arrayBuffer();
      return res.status === 200;
    });
  });

  afterAll(async () => {
    await app?.exit();
    await otlp?.close();
    await otlpWide?.close();
    await Promise.all(Object.values(up).map((u) => u.close()));
  });

  test("reaches a /v1/messages caller as a text block", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const res = await post(app, "/v1/messages", {
      model: "ac-short",
      max_tokens: 64,
      messages: [{ role: "user", content: "What is the capital of France?" }],
    });
    const body = (await res.json()) as any;
    expect(res.status, JSON.stringify(body)).toBe(200);
    expect(body.content).toEqual([{ type: "text", text: ANSWER }]);
    expect(body.stop_reason).toBe("end_turn");
  });

  test("reaches a /v1/responses caller as an output_text message", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const res = await post(app, "/v1/responses", {
      model: "ac-short",
      input: "What is the capital of France?",
    });
    const body = (await res.json()) as any;
    expect(res.status, JSON.stringify(body)).toBe(200);
    expect(body.status, JSON.stringify(body)).toBe("completed");
    const messages = (body.output ?? []).filter((item: any) => item.type === "message");
    expect(messages).toHaveLength(1);
    expect(messages[0].content).toMatchObject([{ type: "output_text", text: ANSWER }]);
  });

  test.for([
    ["/v1/chat/completions", "chat-capture-probe", { messages: [{ role: "user", content: "chat-capture-probe" }] }],
    ["/v1/messages", "messages-capture-probe", { max_tokens: 64, messages: [{ role: "user", content: "messages-capture-probe" }] }],
  ] as const)("reaches the content exporter from %s", async ([path, probe, body], ctx) => {
    if (!etcdReachable || !app || !otlp) return ctx.skip();
    const res = await post(app, path, { model: "ac-long", ...body });
    expect(res.status, await res.text()).toBe(200);
    const [span] = await waitForSpans(otlp, (a) => promptOf(a).includes(probe), 1);
    expect(span["aisix.content.response"]).toBeDefined();
    expect(span["aisix.content.response"].startsWith("array-answer-array-answer-")).toBe(true);
    expect(span["aisix.content_truncated"]).toBe("true");
  });

  test("reaches the content exporter from a response-cache hit", async (ctx) => {
    if (!etcdReachable || !app || !otlp) return ctx.skip();
    const probe = "cache-capture-probe";
    const before = up.cached.receivedRequests.length;
    for (let i = 0; i < 2; i++) {
      const res = await post(app, "/v1/chat/completions", {
        model: "ac-cached",
        messages: [{ role: "user", content: probe }],
      });
      expect(res.status, await res.text()).toBe(200);
    }
    // The second request was answered from the cache.
    expect(up.cached.receivedRequests.length - before).toBe(1);
    const spans = await waitForSpans(otlp, (a) => promptOf(a).includes(probe), 2);
    for (const span of spans) {
      expect(span["aisix.content.response"]).toBeDefined();
      expect(span["aisix.content.response"].startsWith("array-answer-array-answer-")).toBe(true);
    }
  });

  test.for(["mistral", "databricks"] as const)(
    "reasoning parts (%s) reach a /v1/messages caller as a thinking block ahead of the text",
    async (shape, ctx) => {
      if (!etcdReachable || !app) return ctx.skip();
      const res = await post(app, "/v1/messages", {
        model: `ac-${shape}`,
        max_tokens: 64,
        messages: [{ role: "user", content: "What is the capital of France?" }],
      });
      const body = (await res.json()) as any;
      expect(res.status, JSON.stringify(body)).toBe(200);
      expect(body.content).toEqual([
        { type: "thinking", thinking: THINKING, signature: "" },
        { type: "text", text: ANSWER },
      ]);
    },
  );

  test.for(["mistral", "databricks"] as const)(
    "reasoning parts (%s) reach a /v1/responses caller as a reasoning item ahead of the message",
    async (shape, ctx) => {
      if (!etcdReachable || !app) return ctx.skip();
      const res = await post(app, "/v1/responses", { model: `ac-${shape}`, input: "What is the capital of France?" });
      const body = (await res.json()) as any;
      expect(res.status, JSON.stringify(body)).toBe(200);
      expect(body.output.map((item: any) => item.type)).toEqual(["reasoning", "message"]);
      expect(body.output[0].summary).toEqual([{ type: "summary_text", text: THINKING }]);
      expect(body.output[1].content).toMatchObject([{ type: "output_text", text: ANSWER }]);
    },
  );

  test("reasoning parts count toward the completion tokens estimated for an answer without usage", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const res = await post(app, "/v1/chat/completions", {
      model: "ac-no-usage",
      messages: [{ role: "user", content: "What is the capital of France?" }],
    });
    const body = (await res.json()) as any;
    expect(res.status, JSON.stringify(body)).toBe(200);
    // The answer text alone is under ten tokens; the reasoning is hundreds.
    expect(body.usage.completion_tokens, JSON.stringify(body.usage)).toBeGreaterThan(100);
  });

  test("reasoning parts reach the content exporter as a reasoning part", async (ctx) => {
    if (!etcdReachable || !app || !otlpWide) return ctx.skip();
    const probe = "reasoning-capture-probe";
    const res = await post(app, "/v1/chat/completions", {
      model: "ac-mistral",
      messages: [{ role: "user", content: probe }],
    });
    expect(res.status, await res.text()).toBe(200);
    const [span] = await waitForSpans(otlpWide, (a) => promptOf(a).includes(probe), 1);
    const output = JSON.parse(span["gen_ai.output.messages"] ?? "null");
    expect(output?.[0]?.parts, span["gen_ai.output.messages"]).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ type: "reasoning", content: THINKING }),
        expect.objectContaining({ type: "text", content: ANSWER }),
      ]),
    );
  });

  test("a response-cache hit reaches the content exporter with the output messages of the answer it replays", async (ctx) => {
    if (!etcdReachable || !app || !otlpWide) return ctx.skip();
    const probe = "reasoning-cache-capture-probe";
    const before = up.mistralCached.receivedRequests.length;
    const statuses: string[] = [];
    for (let i = 0; i < 2; i++) {
      const res = await post(app, "/v1/chat/completions", {
        model: "ac-mistral-cached",
        messages: [{ role: "user", content: probe }],
      });
      expect(res.status, await res.text()).toBe(200);
      statuses.push(res.headers.get("x-aisix-cache") ?? "");
    }
    expect(statuses).toEqual(["miss", "hit"]);
    expect(up.mistralCached.receivedRequests.length - before).toBe(1);
    const spans = await waitForSpans(otlpWide, (a) => promptOf(a).includes(probe), 2);
    // Only the answer the upstream produced carries its response id.
    const fresh = spans.find((a) => a["gen_ai.response.id"] !== undefined)!;
    const hit = spans.find((a) => a["gen_ai.response.id"] === undefined)!;
    expect(fresh && hit, JSON.stringify(spans)).toBeTruthy();
    const freshOutput = JSON.parse(fresh["gen_ai.output.messages"] ?? "null");
    expect(freshOutput?.[0]?.parts, fresh["gen_ai.output.messages"]).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ type: "reasoning", content: THINKING }),
        expect.objectContaining({ type: "text", content: ANSWER }),
      ]),
    );
    expect(JSON.parse(hit["gen_ai.output.messages"] ?? "null"), JSON.stringify(hit)).toEqual(freshOutput);
    expect(fresh["gen_ai.response.finish_reasons"]).toBe('["stop"]');
    expect(hit["gen_ai.response.finish_reasons"]).toBe('["stop"]');
  });
});
