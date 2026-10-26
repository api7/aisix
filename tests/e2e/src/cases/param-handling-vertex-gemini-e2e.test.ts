import { createHash } from "node:crypto";
import { createServer, type IncomingMessage, type Server } from "node:http";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import { EtcdClient, SeedClient, spawnApp, waitConfigPropagation, type SpawnedApp } from "../harness/index.js";

// Vertex's native Gemini wire (`:generateContent` / `:streamGenerateContent`)
// handles each request field by Gemini's contract. OpenAI sampling fields reach
// their `generationConfig` slots as sent, `reasoning_effort` becomes each
// generation's thinking setting, `service_tier` the PayGo tier header, and an
// `allowed_tools` subset a restricted forced call. Every candidate comes back
// as a choice with its logprobs, and a blocked prompt finishes as
// `content_filter`. A `custom` tool has no Gemini declaration and is dropped,
// while a value OpenAI's own contract forbids is refused naming the field.

const CALLER = "sk-param-vertex-gemini";
const CALLER_HASH = createHash("sha256").update(CALLER).digest("hex");

type Received = { path: string; headers: IncomingMessage["headers"]; body: any };
type Reply = { json?: unknown; sse?: unknown[] };

const USAGE = { promptTokenCount: 5, candidatesTokenCount: 2, totalTokenCount: 7 };
const TEXT_REPLY = {
  candidates: [{ content: { role: "model", parts: [{ text: "ok" }] }, finishReason: "STOP" }],
  usageMetadata: USAGE,
};
const TEXT_FRAMES = [TEXT_REPLY];

function logprobsResult(token: string) {
  return {
    chosenCandidates: [{ token, logProbability: -0.25 }],
    topCandidates: [{ candidates: [{ token, logProbability: -0.25 }, { token: "z", logProbability: -3 }] }],
  };
}

function openAiLogprobs(token: string) {
  const bytes = [...Buffer.from(token, "utf8")];
  return {
    content: [
      {
        token,
        logprob: -0.25,
        bytes,
        top_logprobs: [
          { token, logprob: -0.25, bytes },
          { token: "z", logprob: -3, bytes: [122] },
        ],
      },
    ],
  };
}

const WEATHER = {
  type: "function",
  function: {
    name: "get_weather",
    parameters: {
      type: "object",
      properties: { city: { type: "string" } },
      required: ["city"],
      additionalProperties: false,
    },
  },
};
const NOW = { type: "function", function: { name: "now" } };
const HI = [{ role: "user", content: "hi" }];

describe("Vertex Gemini: request fields by the target's contract", () => {
  let app: SpawnedApp | undefined;
  let upstream: Server | undefined;
  let etcdReachable = false;
  const received: Received[] = [];
  let respond: (r: Received) => Reply = () => ({ json: TEXT_REPLY, sse: TEXT_FRAMES });

  async function post(path: string, body: Record<string, unknown>) {
    const res = await fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER}`,
        "x-api-key": CALLER,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
      },
      body: JSON.stringify(body),
    });
    return { status: res.status, text: await res.text() };
  }

  /** Run `fn` and return its result with the upstream requests it caused. */
  async function capture<T>(fn: () => Promise<T>): Promise<{ result: T; sent: Received[] }> {
    const before = received.length;
    const result = await fn();
    return { result, sent: received.slice(before) };
  }

  const chat = (body: Record<string, unknown>) => post("/v1/chat/completions", body);
  const skip = (ctx: { skip: () => void }) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return true;
    }
    return false;
  };
  const sseData = (raw: string): any[] =>
    raw
      .split(/\r?\n/)
      .filter((l) => l.startsWith("data: ") && !l.includes("[DONE]"))
      .map((l) => JSON.parse(l.slice(6)));

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = createServer((req, res) => {
      const chunks: Buffer[] = [];
      req.on("data", (c: Buffer) => chunks.push(c));
      req.on("end", () => {
        const r: Received = {
          path: (req.url ?? "/").split("?")[0],
          headers: req.headers,
          body: JSON.parse(Buffer.concat(chunks).toString("utf8") || "{}"),
        };
        received.push(r);
        const reply = respond(r);
        res.statusCode = 200;
        if (r.path.endsWith(":streamGenerateContent")) {
          res.setHeader("content-type", "text/event-stream");
          res.end((reply.sse ?? []).map((f) => `data: ${JSON.stringify(f)}\r\n\r\n`).join(""));
          return;
        }
        res.setHeader("content-type", "application/json");
        res.end(JSON.stringify(reply.json));
      });
    });
    await new Promise<void>((resolve) => upstream!.listen(0, "127.0.0.1", resolve));
    const addr = upstream.address();
    if (addr === null || typeof addr === "string") throw new Error("no port");

    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);
    const pk = await seed.createProviderKey({
      display_name: "param-vertex-gemini-pk",
      provider: "google",
      adapter: "vertex",
      secret: JSON.stringify({ access_token: "ya29.param", project: "proj-param", region: "us-central1" }),
      api_base: `http://127.0.0.1:${addr.port}`,
    });
    const models: Array<[string, string]> = [
      ["pg-25-flash", "gemini-2.5-flash"],
      ["pg-25-pro", "gemini-2.5-pro"],
      ["pg-3-flash", "gemini-3-flash-preview"],
      ["pg-20-flash", "gemini-2.0-flash-001"],
    ];
    for (const [alias, name] of models) {
      await seed.createModel({ display_name: alias, provider: "google", model_name: name, provider_key_id: pk.id });
    }
    await seed.createApiKey({ key_hash: CALLER_HASH, allowed_models: ["*"] });
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, { headers: { authorization: `Bearer ${CALLER}` } });
      await res.text();
      return res.status === 200;
    });
  });

  afterAll(async () => {
    await app?.exit();
    await new Promise<void>((resolve) => (upstream ? upstream.close(() => resolve()) : resolve()));
  });

  test("OpenAI sampling fields reach their generationConfig slots as sent", async (ctx) => {
    if (skip(ctx)) return;
    const { result, sent } = await capture(() =>
      chat({
        model: "pg-25-flash",
        messages: HI,
        max_tokens: 10,
        max_completion_tokens: 20,
        stop: "END",
        seed: 7,
        presence_penalty: 0.5,
        frequency_penalty: -0.25,
      }),
    );
    expect(result.status, result.text).toBe(200);
    expect(sent[0].body.generationConfig).toEqual({
      maxOutputTokens: 20,
      stopSequences: ["END"],
      seed: 7,
      presencePenalty: 0.5,
      frequencyPenalty: -0.25,
    });
  });

  test("reasoning_effort becomes thinkingLevel on Gemini 3 and the published budget on 2.5", async (ctx) => {
    if (skip(ctx)) return;
    const cases: Array<[string, string, unknown]> = [
      ["pg-3-flash", "minimal", { thinkingLevel: "MINIMAL" }],
      ["pg-3-flash", "high", { thinkingLevel: "HIGH" }],
      ["pg-25-flash", "low", { thinkingBudget: 1024 }],
      ["pg-25-flash", "medium", { thinkingBudget: 8192 }],
      ["pg-25-pro", "high", { thinkingBudget: 24576 }],
      ["pg-25-flash", "none", { thinkingBudget: 0 }],
      // 2.5 Pro cannot turn thinking off, and 2.0 has no thinking setting.
      ["pg-25-pro", "none", undefined],
      ["pg-20-flash", "high", undefined],
    ];
    for (const [model, effort, expected] of cases) {
      const { result, sent } = await capture(() => chat({ model, messages: HI, reasoning_effort: effort }));
      expect(result.status, `${model} ${effort}: ${result.text}`).toBe(200);
      expect(sent[0].body.generationConfig?.thinkingConfig, `${model} ${effort}`).toEqual(expected);
    }
  });

  test("/v1/messages top_k, stop_sequences and the exact thinking budget reach Gemini", async (ctx) => {
    if (skip(ctx)) return;
    const { result, sent } = await capture(() =>
      post("/v1/messages", {
        model: "pg-25-flash",
        max_tokens: 4096,
        top_k: 40,
        stop_sequences: ["END"],
        thinking: { type: "enabled", budget_tokens: 3000 },
        messages: HI,
      }),
    );
    expect(result.status, result.text).toBe(200);
    expect(sent[0].body.generationConfig).toMatchObject({
      topK: 40,
      stopSequences: ["END"],
      maxOutputTokens: 4096,
      thinkingConfig: { thinkingBudget: 3000 },
    });
  });

  test("n and logprobs come back as every choice with its logprobs", async (ctx) => {
    if (skip(ctx)) return;
    respond = () => ({
      json: {
        candidates: [
          {
            content: { role: "model", parts: [{ text: "A" }] },
            finishReason: "STOP",
            logprobsResult: logprobsResult("A"),
          },
          {
            index: 1,
            content: { role: "model", parts: [{ text: "B" }] },
            finishReason: "MAX_TOKENS",
            logprobsResult: logprobsResult("B"),
          },
        ],
        usageMetadata: USAGE,
      },
      sse: [
        {
          candidates: [
            { content: { role: "model", parts: [{ text: "A" }] }, logprobsResult: logprobsResult("A") },
            { index: 1, content: { role: "model", parts: [{ text: "B" }] }, logprobsResult: logprobsResult("B") },
          ],
        },
        {
          candidates: [{ finishReason: "STOP" }, { index: 1, finishReason: "MAX_TOKENS" }],
          usageMetadata: USAGE,
        },
      ],
    });
    try {
      const request = { model: "pg-25-flash", messages: HI, n: 2, logprobs: true, top_logprobs: 2 };
      const { result, sent } = await capture(() => chat(request));
      expect(result.status, result.text).toBe(200);
      expect(sent[0].body.generationConfig).toEqual({ candidateCount: 2, responseLogprobs: true, logprobs: 2 });
      const body = JSON.parse(result.text);
      expect(
        body.choices.map((c: any) => [c.index, c.message.content, c.finish_reason, c.logprobs]),
      ).toEqual([
        [0, "A", "stop", openAiLogprobs("A")],
        [1, "B", "length", openAiLogprobs("B")],
      ]);

      const streamed = await chat({ ...request, stream: true });
      expect(streamed.status, streamed.text).toBe(200);
      const choices = sseData(streamed.text).flatMap((f) => f.choices ?? []);
      for (const [index, text, finish] of [
        [0, "A", "stop"],
        [1, "B", "length"],
      ] as const) {
        const mine = choices.filter((c: any) => c.index === index);
        expect(mine.map((c: any) => c.delta?.content ?? "").join(""), `choice ${index}`).toBe(text);
        expect(mine.find((c: any) => c.logprobs)?.logprobs, `choice ${index}`).toEqual(openAiLogprobs(text));
        expect(mine.find((c: any) => c.finish_reason)?.finish_reason, `choice ${index}`).toBe(finish);
      }
    } finally {
      respond = () => ({ json: TEXT_REPLY, sse: TEXT_FRAMES });
    }
  });

  test("a blocked prompt finishes as content_filter", async (ctx) => {
    if (skip(ctx)) return;
    const blocked = {
      promptFeedback: { blockReason: "PROHIBITED_CONTENT" },
      usageMetadata: { promptTokenCount: 5, totalTokenCount: 5 },
    };
    respond = () => ({ json: blocked, sse: [blocked] });
    try {
      const res = await chat({ model: "pg-25-flash", messages: HI });
      expect(res.status, res.text).toBe(200);
      expect(JSON.parse(res.text).choices[0].finish_reason).toBe("content_filter");

      const streamed = await chat({ model: "pg-25-flash", messages: HI, stream: true });
      expect(streamed.status, streamed.text).toBe(200);
      const finishes = sseData(streamed.text)
        .flatMap((f) => f.choices ?? [])
        .map((c: any) => c.finish_reason)
        .filter(Boolean);
      expect(finishes).toEqual(["content_filter"]);
    } finally {
      respond = () => ({ json: TEXT_REPLY, sse: TEXT_FRAMES });
    }
  });

  test("service_tier flex and priority pick the Vertex PayGo tier header", async (ctx) => {
    if (skip(ctx)) return;
    for (const [tier, header] of [
      ["flex", "flex"],
      ["priority", "priority"],
      ["auto", undefined],
      ["default", undefined],
    ] as const) {
      for (const stream of [false, true]) {
        const { result, sent } = await capture(() =>
          chat({ model: "pg-25-flash", messages: HI, service_tier: tier, stream }),
        );
        expect(result.status, `${tier} stream=${stream}: ${result.text}`).toBe(200);
        expect(sent[0].headers["x-vertex-ai-llm-shared-request-type"], `${tier} stream=${stream}`).toBe(header);
        expect(sent[0].body.service_tier).toBeUndefined();
      }
    }
  });

  test("an allowed_tools subset forces a call among its functions in required mode", async (ctx) => {
    if (skip(ctx)) return;
    const allowed = (mode: string) => ({
      type: "allowed_tools",
      allowed_tools: { mode, tools: [{ type: "function", function: { name: "now" } }] },
    });
    let { result, sent } = await capture(() =>
      chat({ model: "pg-25-flash", messages: HI, tools: [WEATHER, NOW], tool_choice: allowed("required") }),
    );
    expect(result.status, result.text).toBe(200);
    expect(sent[0].body.toolConfig).toEqual({
      functionCallingConfig: { mode: "ANY", allowedFunctionNames: ["now"] },
    });
    ({ result, sent } = await capture(() =>
      chat({ model: "pg-25-flash", messages: HI, tools: [WEATHER, NOW], tool_choice: allowed("auto") }),
    ));
    expect(result.status, result.text).toBe(200);
    expect(sent[0].body.toolConfig).toBeUndefined();
  });

  test("function parameters go verbatim as parametersJsonSchema on 2.5 and later, typed before", async (ctx) => {
    if (skip(ctx)) return;
    for (const model of ["pg-25-flash", "pg-25-pro", "pg-3-flash"]) {
      const { result, sent } = await capture(() => chat({ model, messages: HI, tools: [WEATHER] }));
      expect(result.status, `${model}: ${result.text}`).toBe(200);
      expect(sent[0].body.tools[0].functionDeclarations[0], model).toEqual({
        name: "get_weather",
        parametersJsonSchema: WEATHER.function.parameters,
      });
    }
    const { result, sent } = await capture(() => chat({ model: "pg-20-flash", messages: HI, tools: [WEATHER] }));
    expect(result.status, result.text).toBe(200);
    expect(sent[0].body.tools[0].functionDeclarations[0]).toEqual({
      name: "get_weather",
      parameters: { type: "object", properties: { city: { type: "string" } }, required: ["city"] },
    });
  });

  test("a custom tool is dropped, not refused", async (ctx) => {
    if (skip(ctx)) return;
    const { result, sent } = await capture(() =>
      chat({
        model: "pg-25-flash",
        messages: HI,
        tools: [{ type: "custom", custom: { name: "grammar" } }, NOW],
      }),
    );
    expect(result.status, result.text).toBe(200);
    expect(sent[0].body.tools).toEqual([{ functionDeclarations: [{ name: "now" }] }]);
  });

  test.for<[string, Record<string, unknown>, string]>([
    ["tools that are not an array", { tools: { type: "function" } }, "tools must be an array"],
    ["a numeric tool_choice", { tools: [NOW], tool_choice: 1 }, "tool_choice must be"],
    ["a boolean tool_choice without tools", { tool_choice: true }, "tool_choice must be"],
    ["a string response_format", { response_format: "json" }, "response_format must be an object"],
    ["an unknown response_format type", { response_format: { type: "yaml" } }, "response_format.type must be"],
  ])("%s is refused naming the field", async ([, extra, message], ctx) => {
    if (skip(ctx)) return;
    for (const stream of [false, true]) {
      const { result, sent } = await capture(() => chat({ model: "pg-25-flash", messages: HI, stream, ...extra }));
      expect(result.status, `stream=${stream}: ${result.text}`).toBe(400);
      expect(JSON.parse(result.text).error.message, `stream=${stream}`).toContain(message);
      expect(sent, `stream=${stream}`).toHaveLength(0);
    }
  });

  test("a null tool_choice stays unset", async (ctx) => {
    if (skip(ctx)) return;
    const { result, sent } = await capture(() => chat({ model: "pg-25-flash", messages: HI, tools: [NOW], tool_choice: null }));
    expect(result.status, result.text).toBe(200);
    expect(sent[0].body.toolConfig).toBeUndefined();
  });
});
