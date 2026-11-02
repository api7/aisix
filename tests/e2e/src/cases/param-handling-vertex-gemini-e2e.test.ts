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
//
// A caller that asks for thinking also gets `includeThoughts`, and `thought`
// parts reach every client protocol as its own reasoning field, never as answer
// text. The frames below are the shapes real Vertex returned for
// `gemini-2.5-flash` and `gemini-3.8-flash`.

const CALLER = "sk-param-vertex-gemini";
const CALLER_HASH = createHash("sha256").update(CALLER).digest("hex");

type Received = { path: string; headers: IncomingMessage["headers"]; body: any };
type Reply = { json?: unknown; sse?: unknown[]; holdRestUntil?: Promise<void> };

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

const THOUGHT = "**Formulating Solution**\n\n  I'm currently working it out.\n";
const ANSWER = "The ball costs 0.05.";
const SIGNATURE = "c2lnbmF0dXJl".repeat(200);
const THINKING_USAGE = { promptTokenCount: 37, candidatesTokenCount: 233, totalTokenCount: 889, thoughtsTokenCount: 619 };

const frame = (parts: unknown[], rest: Record<string, unknown> = {}) => ({
  candidates: [{ content: { role: "model", parts }, ...rest }],
  usageMetadata: rest.finishReason ? THINKING_USAGE : { trafficType: "ON_DEMAND" },
});
const THOUGHT_PART = { text: THOUGHT, thought: true };
const WEATHER_CALL = { functionCall: { name: "get_weather", args: { city: "Paris" } }, thoughtSignature: SIGNATURE };

// Gemini 3: the answer part is signed, and the last stream frame is empty text
// plus a signature. Gemini 2.5: no signature anywhere.
const THINKING_REPLY = {
  json: {
    candidates: [
      {
        content: { role: "model", parts: [THOUGHT_PART, { text: ANSWER, thoughtSignature: SIGNATURE }] },
        finishReason: "STOP",
      },
    ],
    usageMetadata: THINKING_USAGE,
  },
  sse: [
    frame([THOUGHT_PART]),
    frame([{ text: ANSWER }]),
    frame([{ text: "", thoughtSignature: SIGNATURE }], { finishReason: "STOP" }),
  ],
};
const THINKING_REPLY_25 = {
  json: {
    candidates: [{ content: { role: "model", parts: [THOUGHT_PART, { text: ANSWER }] }, finishReason: "STOP" }],
    usageMetadata: THINKING_USAGE,
  },
  sse: [frame([THOUGHT_PART]), frame([{ text: ANSWER }]), frame([{ text: "" }], { finishReason: "STOP" })],
};
const THOUGHT_THEN_CALL_REPLY = {
  json: {
    candidates: [{ content: { role: "model", parts: [THOUGHT_PART, WEATHER_CALL] }, finishReason: "STOP" }],
    usageMetadata: THINKING_USAGE,
  },
  sse: [frame([THOUGHT_PART]), frame([WEATHER_CALL]), frame([{ text: "" }], { finishReason: "STOP" })],
};

const ANTHROPIC_WEATHER = {
  name: "get_weather",
  input_schema: WEATHER.function.parameters,
};
const RESPONSES_WEATHER = { type: "function", name: "get_weather", parameters: WEATHER.function.parameters };

type Surface = {
  path: string;
  request: (model: string) => Record<string, unknown>;
  tools: Record<string, unknown>;
  /** Pull reasoning, answer text and tool-call names out of a whole response. */
  read: (body: any) => { reasoning: string; answer: string; calls: string[] };
  /** The same from a stream, in event order; `seen` lists what arrived, in order. */
  readStream: (raw: string) => { reasoning: string; answer: string; calls: string[]; seen: string[] };
  reasoningOnTheWire: string;
  /** The next request's history, replaying `body` and answering its tool call. */
  replayToolTurn: (body: any) => Record<string, unknown>;
};

const sseEvents = (raw: string): any[] =>
  raw
    .split(/\r?\n/)
    .filter((l) => l.startsWith("data: ") && !l.includes("[DONE]"))
    .map((l) => JSON.parse(l.slice(6)));

const SURFACES: Array<[string, Surface]> = [
  [
    "/v1/chat/completions",
    {
      path: "/v1/chat/completions",
      request: (model) => ({ model, messages: HI, reasoning_effort: "high" }),
      tools: { tools: [WEATHER] },
      read: (body) => {
        const message = body.choices[0].message;
        return {
          reasoning: message.reasoning_content ?? "",
          answer: message.content ?? "",
          calls: (message.tool_calls ?? []).map((c: any) => c.function.name),
        };
      },
      readStream: (raw) => {
        const out = { reasoning: "", answer: "", calls: [] as string[], seen: [] as string[] };
        for (const e of sseEvents(raw)) {
          const delta = e.choices?.[0]?.delta ?? {};
          if (delta.reasoning_content) {
            out.reasoning += delta.reasoning_content;
            out.seen.push("reasoning");
          }
          if (delta.content) {
            out.answer += delta.content;
            out.seen.push("answer");
          }
          for (const call of delta.tool_calls ?? []) {
            out.calls.push(call.function.name);
            out.seen.push("call");
          }
        }
        return out;
      },
      reasoningOnTheWire: '"reasoning_content":',
      replayToolTurn: (body) => {
        const message = body.choices[0].message;
        return {
          messages: [...HI, message, { role: "tool", tool_call_id: message.tool_calls[0].id, content: "sunny" }],
        };
      },
    },
  ],
  [
    "/v1/messages",
    {
      path: "/v1/messages",
      request: (model) => ({
        model,
        max_tokens: 4096,
        messages: HI,
        thinking: { type: "enabled", budget_tokens: 3000 },
      }),
      tools: { tools: [ANTHROPIC_WEATHER] },
      read: (body) => ({
        reasoning: body.content
          .filter((b: any) => b.type === "thinking")
          .map((b: any) => b.thinking)
          .join(""),
        answer: body.content
          .filter((b: any) => b.type === "text")
          .map((b: any) => b.text)
          .join(""),
        calls: body.content.filter((b: any) => b.type === "tool_use").map((b: any) => b.name),
      }),
      readStream: (raw) => {
        const out = { reasoning: "", answer: "", calls: [] as string[], seen: [] as string[] };
        for (const e of sseEvents(raw)) {
          if (e.type === "content_block_delta" && e.delta.type === "thinking_delta") {
            out.reasoning += e.delta.thinking;
            out.seen.push("reasoning");
          }
          if (e.type === "content_block_delta" && e.delta.type === "text_delta") {
            out.answer += e.delta.text;
            out.seen.push("answer");
          }
          if (e.type === "content_block_start" && e.content_block.type === "tool_use") {
            out.calls.push(e.content_block.name);
            out.seen.push("call");
          }
        }
        return out;
      },
      reasoningOnTheWire: '"thinking_delta"',
      replayToolTurn: (body) => ({
        messages: [
          ...HI,
          { role: "assistant", content: body.content },
          {
            role: "user",
            content: [
              {
                type: "tool_result",
                tool_use_id: body.content.find((b: any) => b.type === "tool_use").id,
                content: "sunny",
              },
            ],
          },
        ],
      }),
    },
  ],
  [
    "/v1/responses",
    {
      path: "/v1/responses",
      request: (model) => ({ model, input: "hi", reasoning: { effort: "high" } }),
      tools: { tools: [RESPONSES_WEATHER] },
      read: (body) => ({
        reasoning: body.output
          .filter((o: any) => o.type === "reasoning")
          .flatMap((o: any) => o.summary.map((s: any) => s.text))
          .join(""),
        answer: body.output
          .filter((o: any) => o.type === "message")
          .flatMap((o: any) => o.content.map((c: any) => c.text))
          .join(""),
        calls: body.output.filter((o: any) => o.type === "function_call").map((o: any) => o.name),
      }),
      readStream: (raw) => {
        const out = { reasoning: "", answer: "", calls: [] as string[], seen: [] as string[] };
        for (const e of sseEvents(raw)) {
          if (e.type === "response.reasoning_summary_text.delta") {
            out.reasoning += e.delta;
            out.seen.push("reasoning");
          }
          if (e.type === "response.output_text.delta") {
            out.answer += e.delta;
            out.seen.push("answer");
          }
          if (e.type === "response.output_item.added" && e.item.type === "function_call") {
            out.calls.push(e.item.name);
            out.seen.push("call");
          }
        }
        return out;
      },
      reasoningOnTheWire: '"response.reasoning_summary_text.delta"',
      replayToolTurn: (body) => ({
        input: [
          { role: "user", content: "hi" },
          ...body.output,
          {
            type: "function_call_output",
            call_id: body.output.find((o: any) => o.type === "function_call").call_id,
            output: "sunny",
          },
        ],
      }),
    },
  ],
];

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

  /**
   * Stream `body` while the upstream sends only its first frame and holds the
   * rest until the client has read `firstEvent`, so a gateway that waits for the
   * answer before forwarding reasoning times out here.
   */
  async function streamHoldingTheRest(path: string, body: Record<string, unknown>, reply: Reply, firstEvent: string) {
    let release = () => {};
    const released = new Promise<void>((resolve) => (release = resolve));
    respond = () => ({ ...reply, holdRestUntil: released });
    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      release();
    }, 5000);
    let sawFirstEventBeforeRest = false;
    try {
      const res = await fetch(`${app!.proxyUrl}${path}`, {
        method: "POST",
        headers: {
          authorization: `Bearer ${CALLER}`,
          "x-api-key": CALLER,
          "anthropic-version": "2023-06-01",
          "content-type": "application/json",
        },
        body: JSON.stringify({ ...body, stream: true }),
      });
      expect(res.status).toBe(200);
      let raw = "";
      const decoder = new TextDecoder();
      for await (const piece of res.body as unknown as AsyncIterable<Uint8Array>) {
        raw += decoder.decode(piece, { stream: true });
        if (!sawFirstEventBeforeRest && raw.includes(firstEvent)) {
          sawFirstEventBeforeRest = !timedOut;
          release();
        }
      }
      return { raw, sawFirstEventBeforeRest };
    } finally {
      clearTimeout(timer);
      release();
      respond = () => ({ json: TEXT_REPLY, sse: TEXT_FRAMES });
    }
  }
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
      req.on("end", async () => {
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
          const data = (reply.sse ?? []).map((f) => `data: ${JSON.stringify(f)}\r\n\r\n`);
          if (reply.holdRestUntil) {
            res.write(data[0]);
            await reply.holdRestUntil;
            res.end(data.slice(1).join(""));
            return;
          }
          res.end(data.join(""));
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
      ["pg-3-flash", "minimal", { thinkingLevel: "MINIMAL", includeThoughts: true }],
      ["pg-3-flash", "high", { thinkingLevel: "HIGH", includeThoughts: true }],
      ["pg-25-flash", "low", { thinkingBudget: 1024, includeThoughts: true }],
      ["pg-25-flash", "medium", { thinkingBudget: 8192, includeThoughts: true }],
      ["pg-25-pro", "high", { thinkingBudget: 24576, includeThoughts: true }],
      // A tier with no setting of its own still asked for thinking.
      ["pg-3-flash", "max", { includeThoughts: true }],
      ["pg-25-flash", "max", { includeThoughts: true }],
      // `none` asks for no thoughts.
      ["pg-25-flash", "none", { thinkingBudget: 0 }],
      ["pg-3-flash", "none", undefined],
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
    });
    expect(sent[0].body.generationConfig.thinkingConfig).toEqual({ thinkingBudget: 3000, includeThoughts: true });
  });

  test.for<[string, Record<string, unknown>, unknown]>([
    ["enabled budget", { thinking: { type: "enabled", budget_tokens: 3000 } }, { thinkingLevel: "MEDIUM", includeThoughts: true }],
    ["adaptive", { thinking: { type: "adaptive" } }, { thinkingLevel: "HIGH", includeThoughts: true }],
    ["adaptive with an effort", { thinking: { type: "adaptive" }, output_config: { effort: "low" } }, { thinkingLevel: "LOW", includeThoughts: true }],
    ["disabled", { thinking: { type: "disabled" } }, undefined],
    ["disabled with an effort", { thinking: { type: "disabled" }, output_config: { effort: "high" } }, undefined],
    ["nothing", {}, undefined],
  ])("/v1/messages thinking on Gemini 3: %s", async ([label, extra, expected], ctx) => {
    if (skip(ctx)) return;
    const { result, sent } = await capture(() =>
      post("/v1/messages", { model: "pg-3-flash", max_tokens: 4096, messages: HI, ...extra }),
    );
    expect(result.status, result.text).toBe(200);
    expect(sent[0].body.generationConfig?.thinkingConfig, label).toEqual(expected);
  });

  test("/v1/messages turning thinking off or sending nothing keeps the 2.5 request as it was", async (ctx) => {
    if (skip(ctx)) return;
    for (const [extra, expected] of [
      [{ thinking: { type: "disabled" } }, { thinkingBudget: 0 }],
      [{}, undefined],
    ] as const) {
      const { result, sent } = await capture(() =>
        post("/v1/messages", { model: "pg-25-flash", max_tokens: 4096, messages: HI, ...extra }),
      );
      expect(result.status, result.text).toBe(200);
      expect(sent[0].body.generationConfig?.thinkingConfig).toEqual(expected);
    }
  });

  test.for<[string, Record<string, unknown>, unknown]>([
    ["pg-3-flash", { reasoning: { effort: "high" } }, { thinkingLevel: "HIGH", includeThoughts: true }],
    ["pg-25-flash", { reasoning: { effort: "low" } }, { thinkingBudget: 1024, includeThoughts: true }],
    ["pg-25-flash", { reasoning: { effort: "none" } }, { thinkingBudget: 0 }],
    ["pg-3-flash", { reasoning: { effort: "none" } }, undefined],
    ["pg-25-flash", {}, undefined],
    ["pg-3-flash", {}, undefined],
    ["pg-20-flash", { reasoning: { effort: "high" } }, undefined],
  ])("/v1/responses on %s with %j", async ([model, extra, expected], ctx) => {
    if (skip(ctx)) return;
    const { result, sent } = await capture(() => post("/v1/responses", { model, input: "hi", ...extra }));
    expect(result.status, result.text).toBe(200);
    expect(sent[0].body.generationConfig?.thinkingConfig).toEqual(expected);
  });

  test("/v1/chat/completions with no reasoning_effort sends no thinking setting", async (ctx) => {
    if (skip(ctx)) return;
    for (const model of ["pg-25-flash", "pg-3-flash"]) {
      const { result, sent } = await capture(() => chat({ model, messages: HI }));
      expect(result.status, result.text).toBe(200);
      expect(sent[0].body.generationConfig?.thinkingConfig, model).toBeUndefined();
    }
  });

  describe.each(SURFACES)("thought parts on %s", (_name, surface) => {
    const withReply = async <T>(reply: Reply, fn: () => Promise<T>) => {
      respond = () => reply;
      try {
        return await fn();
      } finally {
        respond = () => ({ json: TEXT_REPLY, sse: TEXT_FRAMES });
      }
    };

    test.for<[string, string, Reply]>([
      ["Gemini 3", "pg-3-flash", THINKING_REPLY],
      ["Gemini 2.5", "pg-25-flash", THINKING_REPLY_25],
    ])("%s non-streaming: reasoning on its own field, out of the answer", async ([, model, reply], ctx) => {
      if (skip(ctx)) return;
      const res = await withReply(reply, () => post(surface.path, surface.request(model)));
      expect(res.status, res.text).toBe(200);
      expect(surface.read(JSON.parse(res.text))).toEqual({ reasoning: THOUGHT, answer: ANSWER, calls: [] });
    });

    test.for<[string, string, Reply]>([
      ["Gemini 3", "pg-3-flash", THINKING_REPLY],
      ["Gemini 2.5", "pg-25-flash", THINKING_REPLY_25],
    ])("%s streaming: reasoning reaches the client before the upstream sends the answer", async ([, model, reply], ctx) => {
      if (skip(ctx)) return;
      const { raw, sawFirstEventBeforeRest } = await streamHoldingTheRest(
        surface.path,
        surface.request(model),
        reply,
        surface.reasoningOnTheWire,
      );
      expect(sawFirstEventBeforeRest, raw).toBe(true);
      const out = surface.readStream(raw);
      expect(out.reasoning).toBe(THOUGHT);
      expect(out.answer).toBe(ANSWER);
      expect(out.seen).toEqual(["reasoning", "answer"]);
    });

    test("a client that did not ask still never sees a thought part as answer text", async (ctx) => {
      if (skip(ctx)) return;
      const { reasoning_effort: _a, thinking: _b, reasoning: _c, ...plain } = surface.request("pg-3-flash") as any;
      const res = await withReply(THINKING_REPLY, () => post(surface.path, plain));
      expect(res.status, res.text).toBe(200);
      expect(surface.read(JSON.parse(res.text))).toEqual({ reasoning: THOUGHT, answer: ANSWER, calls: [] });
    });

    test("a frame holding both a thought and answer text renders the thought first", async (ctx) => {
      if (skip(ctx)) return;
      const reply = { sse: [frame([THOUGHT_PART, { text: ANSWER }], { finishReason: "STOP" })] };
      const res = await withReply(reply, () => post(surface.path, { ...surface.request("pg-3-flash"), stream: true }));
      expect(res.status, res.text).toBe(200);
      const out = surface.readStream(res.text);
      expect(out.reasoning).toBe(THOUGHT);
      expect(out.answer).toBe(ANSWER);
      expect(out.seen).toEqual(["reasoning", "answer"]);
    });

    test("a thought followed by a tool call carries both", async (ctx) => {
      if (skip(ctx)) return;
      const request = { ...surface.request("pg-3-flash"), ...surface.tools };
      const res = await withReply(THOUGHT_THEN_CALL_REPLY, () => post(surface.path, request));
      expect(res.status, res.text).toBe(200);
      expect(surface.read(JSON.parse(res.text))).toEqual({ reasoning: THOUGHT, answer: "", calls: ["get_weather"] });

      const streamed = await withReply(THOUGHT_THEN_CALL_REPLY, () => post(surface.path, { ...request, stream: true }));
      expect(streamed.status, streamed.text).toBe(200);
      const out = surface.readStream(streamed.text);
      expect(out.reasoning).toBe(THOUGHT);
      expect(out.answer).toBe("");
      expect(out.calls).toEqual(["get_weather"]);
      expect(out.seen).toEqual(["reasoning", "call"]);
    });

    test("replaying a thought and its tool call sends the signed call back without the thought", async (ctx) => {
      if (skip(ctx)) return;
      const request = { ...surface.request("pg-3-flash"), ...surface.tools };
      const first = await withReply(THOUGHT_THEN_CALL_REPLY, () => post(surface.path, request));
      expect(first.status, first.text).toBe(200);

      const { result, sent } = await capture(() =>
        post(surface.path, { ...request, ...surface.replayToolTurn(JSON.parse(first.text)) }),
      );
      expect(result.status, result.text).toBe(200);
      const contents = sent[0].body.contents;
      expect(contents[1].role).toBe("model");
      expect(contents[1].parts).toHaveLength(1);
      expect(contents[1].parts[0]).toMatchObject(WEATHER_CALL);
      expect(JSON.stringify(contents)).not.toContain("Formulating");
    });
  });

  test.for<[string, string, Record<string, unknown>]>([
    [
      "/v1/chat/completions",
      "/v1/chat/completions",
      {
        reasoning_effort: "high",
        messages: [
          ...HI,
          { role: "assistant", content: ANSWER, reasoning_content: THOUGHT },
          { role: "assistant", content: null, reasoning_content: THOUGHT },
          { role: "user", content: "again" },
        ],
      },
    ],
    [
      "/v1/messages",
      "/v1/messages",
      {
        max_tokens: 4096,
        thinking: { type: "enabled", budget_tokens: 3000 },
        messages: [
          ...HI,
          {
            role: "assistant",
            content: [
              { type: "thinking", thinking: THOUGHT, signature: "" },
              { type: "text", text: ANSWER },
            ],
          },
          { role: "user", content: "again" },
        ],
      },
    ],
    [
      "/v1/responses",
      "/v1/responses",
      {
        reasoning: { effort: "high" },
        input: [
          { role: "user", content: "hi" },
          { type: "reasoning", id: "rs_1", summary: [{ type: "summary_text", text: THOUGHT }] },
          { type: "message", role: "assistant", content: [{ type: "output_text", text: ANSWER }] },
          { role: "user", content: "again" },
        ],
      },
    ],
  ])("%s: reasoning replayed in history is not sent back to Gemini", async ([, path, body], ctx) => {
    if (skip(ctx)) return;
    const { result, sent } = await capture(() => post(path, { model: "pg-3-flash", ...body }));
    expect(result.status, result.text).toBe(200);
    expect(sent[0].body.contents).toEqual([
      { role: "user", parts: [{ text: "hi" }] },
      { role: "model", parts: [{ text: ANSWER }] },
      { role: "user", parts: [{ text: "again" }] },
    ]);
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
