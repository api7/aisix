import { createHash } from "node:crypto";
import { createServer, type IncomingMessage, type Server } from "node:http";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  decodedTextFor,
  EtcdClient,
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
const HEADERS = {
  authorization: `Bearer ${CALLER}`,
  "x-api-key": CALLER,
  "anthropic-version": "2023-06-01",
  "content-type": "application/json",
};

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
      headers: HEADERS,
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
        headers: HEADERS,
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
    // An operator's effort_mapping writes or removes the effort; it never asks
    // for the caller's thoughts.
    for (const [alias, name, effort_mapping] of [
      ["pg-3-default-high", "gemini-3-flash-preview", { "": "high" }],
      ["pg-25-default-high", "gemini-2.5-flash", { "": "high" }],
      ["pg-3-remove", "gemini-3-flash-preview", { "*": null }],
      ["pg-25-remove", "gemini-2.5-flash", { "*": null }],
      ["pg-3-map-none", "gemini-3-flash-preview", { "*": "none" }],
      ["pg-25-map-none", "gemini-2.5-flash", { "*": "none" }],
    ] as const) {
      await seed.createModel({
        display_name: alias,
        provider: "google",
        model_name: name,
        provider_key_id: pk.id,
        effort_mapping,
      });
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
      // An empty value asks for nothing.
      ["pg-25-flash", "", undefined],
      ["pg-3-flash", "", undefined],
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
    ["adaptive with summarized display", { thinking: { type: "adaptive", display: "summarized" } }, { thinkingLevel: "HIGH", includeThoughts: true }],
    ["adaptive with omitted display", { thinking: { type: "adaptive", display: "omitted" } }, { thinkingLevel: "HIGH" }],
    ["enabled with omitted display", { thinking: { type: "enabled", budget_tokens: 3000, display: "omitted" } }, { thinkingLevel: "MEDIUM" }],
    ["output_config.effort alone", { output_config: { effort: "high" } }, { thinkingLevel: "HIGH" }],
    ["enabled with a budget of 0", { thinking: { type: "enabled", budget_tokens: 0 } }, { thinkingLevel: "MINIMAL" }],
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
      [{ thinking: { type: "enabled", budget_tokens: 0 } }, { thinkingBudget: 0 }],
      [{ thinking: { type: "enabled", budget_tokens: 3000, display: "omitted" } }, { thinkingBudget: 3000 }],
      [{ thinking: { type: "adaptive", display: "omitted" } }, { thinkingBudget: 24576 }],
      [{ output_config: { effort: "high" } }, { thinkingBudget: 24576 }],
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
    ["pg-3-flash", { reasoning: { summary: "auto" } }, undefined],
    ["pg-20-flash", { reasoning: { effort: "high" } }, undefined],
  ])("/v1/responses on %s with %j", async ([model, extra, expected], ctx) => {
    if (skip(ctx)) return;
    const { result, sent } = await capture(() => post("/v1/responses", { model, input: "hi", ...extra }));
    expect(result.status, result.text).toBe(200);
    expect(sent[0].body.generationConfig?.thinkingConfig).toEqual(expected);
  });

  test("/v1/messages adaptive thinking asks for thoughts on 2.5", async (ctx) => {
    if (skip(ctx)) return;
    const { result, sent } = await capture(() =>
      post("/v1/messages", { model: "pg-25-flash", max_tokens: 4096, messages: HI, thinking: { type: "adaptive" } }),
    );
    expect(result.status, result.text).toBe(200);
    expect(sent[0].body.generationConfig.thinkingConfig).toEqual({ thinkingBudget: 24576, includeThoughts: true });
  });

  test("/v1/messages with a budget of 0 stays off when the mapping removes the effort", async (ctx) => {
    if (skip(ctx)) return;
    const { result, sent } = await capture(() =>
      post("/v1/messages", {
        model: "pg-25-remove",
        max_tokens: 4096,
        messages: HI,
        thinking: { type: "enabled", budget_tokens: 0 },
        output_config: { effort: "high" },
      }),
    );
    expect(result.status, result.text).toBe(200);
    expect(sent[0].body.generationConfig?.thinkingConfig).toBeUndefined();
  });

  test("/v1/messages with an enabled budget the mapping removes still asks", async (ctx) => {
    if (skip(ctx)) return;
    const { result, sent } = await capture(() =>
      post("/v1/messages", {
        model: "pg-25-remove",
        max_tokens: 4096,
        messages: HI,
        thinking: { type: "enabled", budget_tokens: 3000 },
        output_config: { effort: "high" },
      }),
    );
    expect(result.status, result.text).toBe(200);
    expect(sent[0].body.generationConfig.thinkingConfig).toEqual({ includeThoughts: true });
  });

  const REQUEST_ON: Array<[string, (model: string, extra: Record<string, unknown>) => [string, Record<string, unknown>]]> = [
    ["/v1/chat/completions", (model, extra) => ["/v1/chat/completions", { model, messages: HI, ...extra }]],
    ["/v1/messages", (model, extra) => ["/v1/messages", { model, max_tokens: 4096, messages: HI, ...extra }]],
    ["/v1/responses", (model, extra) => ["/v1/responses", { model, input: "hi", ...extra }]],
  ];
  // Each carries an effort of the caller's for the mapping to act on.
  const ASKS: Record<string, Record<string, unknown>> = {
    "/v1/chat/completions": { reasoning_effort: "high" },
    "/v1/messages": { thinking: { type: "adaptive" }, output_config: { effort: "high" } },
    "/v1/responses": { reasoning: { effort: "high" } },
  };

  describe.each(REQUEST_ON)("effort_mapping on %s", (name, request) => {
    test("a streaming caller that asked sends includeThoughts too", async (ctx) => {
      if (skip(ctx)) return;
      const [path, body] = request("pg-3-flash", ASKS[name]);
      const { result, sent } = await capture(() => post(path, { ...body, stream: true }));
      expect(result.status, result.text).toBe(200);
      expect(sent[0].body.generationConfig.thinkingConfig.includeThoughts).toBe(true);
    });

    test.for<[string, unknown]>([
      ["pg-3-default-high", { thinkingLevel: "HIGH" }],
      ["pg-25-default-high", { thinkingBudget: 24576 }],
    ])("a %s default writes the effort for a caller that sent nothing, and no includeThoughts", async ([model, expected], ctx) => {
      if (skip(ctx)) return;
      const [path, body] = request(model, {});
      const { result, sent } = await capture(() => post(path, body));
      expect(result.status, result.text).toBe(200);
      expect(sent[0].body.generationConfig?.thinkingConfig).toEqual(expected);
    });

    test.for(["pg-3-remove", "pg-25-remove"])("%s removing the effort of a caller that asked leaves includeThoughts alone", async (model, ctx) => {
      if (skip(ctx)) return;
      const [path, body] = request(model, ASKS[name]);
      const { result, sent } = await capture(() => post(path, body));
      expect(result.status, result.text).toBe(200);
      expect(sent[0].body.generationConfig?.thinkingConfig).toEqual({ includeThoughts: true });
    });

    test.for<[string, unknown]>([
      ["pg-3-map-none", undefined],
      ["pg-25-map-none", { thinkingBudget: 0 }],
    ])("%s mapping the effort to none sends no includeThoughts", async ([model, expected], ctx) => {
      if (skip(ctx)) return;
      const [path, body] = request(model, ASKS[name]);
      const { result, sent } = await capture(() => post(path, body));
      expect(result.status, result.text).toBe(200);
      expect(sent[0].body.generationConfig?.thinkingConfig).toEqual(expected);
    });
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

      const streamed = await withReply(THINKING_REPLY, () => post(surface.path, { ...plain, stream: true }));
      expect(streamed.status, streamed.text).toBe(200);
      const out = surface.readStream(streamed.text);
      expect(out.reasoning).toBe(THOUGHT);
      expect(out.answer).toBe(ANSWER);
    });

    test("a turn cut off while thinking returns the thought and no answer", async (ctx) => {
      if (skip(ctx)) return;
      const reply = {
        json: {
          candidates: [{ content: { role: "model", parts: [THOUGHT_PART] }, finishReason: "MAX_TOKENS" }],
          usageMetadata: THINKING_USAGE,
        },
        sse: [frame([THOUGHT_PART]), frame([{ text: "" }], { finishReason: "MAX_TOKENS" })],
      };
      const res = await withReply(reply, () => post(surface.path, surface.request("pg-3-flash")));
      expect(res.status, res.text).toBe(200);
      expect(surface.read(JSON.parse(res.text))).toEqual({ reasoning: THOUGHT, answer: "", calls: [] });

      const streamed = await withReply(reply, () => post(surface.path, { ...surface.request("pg-3-flash"), stream: true }));
      expect(streamed.status, streamed.text).toBe(200);
      const out = surface.readStream(streamed.text);
      expect(out.reasoning).toBe(THOUGHT);
      expect(out.seen).toEqual(["reasoning"]);
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

  test.for<[string, Record<string, unknown>]>([
    [
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
  ])("%s: reasoning replayed in history is not sent back to Gemini", async ([path, body], ctx) => {
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

// An image model answers `:generateContent` with `inlineData` parts. Chat
// Completions returns them as `images` beside the text, in the shape other
// image-capable OpenAI-compatible upstreams use; the other two client protocols
// keep only the text. The parts mirror what `gemini-nano-banana-2.1` returned: a
// thought, an unsigned interim thought image, more thought, a signed text part
// and the signed final image.
describe("Vertex Gemini: generated images on chat completions", () => {
  // Every image payload carries a marker, so a leak of any of them is greppable
  // wherever it lands.
  const MARKER = "SU1BR0VNQVJLRVI";
  const image = (id: string, bytes = 256) =>
    `${MARKER}${id}`.padEnd(Math.ceil((bytes * 4) / 3 / 4) * 4, "A");
  const IMAGE_A = image("a");
  const IMAGE_B = image("b");
  const THOUGHT_IMAGE = image("thoughtimage");
  const inline = (data: string, mimeType = "image/png") => ({ inlineData: { mimeType, data } });
  const entry = (data: string, index: number, mime = "image/png") => ({
    type: "image_url",
    image_url: { url: `data:${mime};base64,${data}`, detail: "auto" },
    index,
  });

  const USAGE_METADATA = { promptTokenCount: 5, candidatesTokenCount: 1290, totalTokenCount: 1295 };
  const reply = (parts: unknown[], usageMetadata: unknown, finishReason: string) => ({
    candidates: [{ content: { role: "model", parts }, finishReason }],
    usageMetadata,
  });
  const THOUGHT_TEXT = { text: "Sketching it out.", thought: true };
  const MORE_THOUGHT = { text: " Refining it.", thought: true };
  const SIGNATURE_PART = "c2ln".repeat(50);
  const FULL_PARTS = [
    THOUGHT_TEXT,
    { ...inline(THOUGHT_IMAGE), thought: true },
    MORE_THOUGHT,
    { text: "Here it is.", thoughtSignature: SIGNATURE_PART },
    { ...inline(IMAGE_A, "image/jpeg"), thoughtSignature: SIGNATURE_PART },
    inline("QVVESU8=", "audio/wav"),
  ];
  const FULL_IMAGES = [entry(THOUGHT_IMAGE, 0), entry(IMAGE_A, 1, "image/jpeg")];
  // The same answer the way a stream delivers it: one part per frame, then a
  // last frame with empty text, the finish reason and usage. An answer refused
  // outright has no parts in either form.
  const framesOf = (parts: unknown[], usageMetadata: unknown, finishReason: string) => [
    ...parts.map((part) => ({ candidates: [{ content: { role: "model", parts: [part] } }] })),
    {
      candidates: [{ content: { role: "model", parts: parts.length > 0 ? [{ text: "" }] : [] }, finishReason }],
      usageMetadata,
    },
  ];

  const CAPTURE_CREDENTIAL = "imgmock";
  const LOGSTORE = "image-events";
  const MASKED_EMAIL = "someone@example.com";
  const BLOCKED_WORD = "forbidden-word";

  let app: SpawnedApp | undefined;
  let upstream: Server | undefined;
  let openAiCompatible: OpenAiUpstream | undefined;
  let openAiCompatibleStream: OpenAiUpstream | undefined;
  let sls: MockSls | undefined;
  let otlp: MockOtlp | undefined;
  const received: Received[] = [];
  let parts: unknown[] = FULL_PARTS;
  let usageMetadata: unknown = USAGE_METADATA;
  let finishReason = "STOP";
  let unique = 0;

  const useAnswer = (next: unknown[], usage: unknown = USAGE_METADATA, finish = "STOP") => {
    parts = next;
    usageMetadata = usage;
    finishReason = finish;
  };
  const say = (content: string = `prompt-${++unique}`) => [{ role: "user", content }];

  async function post(path: string, body: Record<string, unknown>) {
    const res = await fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: HEADERS,
      body: JSON.stringify(body),
    });
    return { status: res.status, headers: res.headers, text: await res.text() };
  }
  const chat = async (body: Record<string, unknown>) => {
    const res = await post("/v1/chat/completions", body);
    expect(res.status, res.text.slice(0, 500)).toBe(200);
    return JSON.parse(res.text).choices[0].message;
  };
  const chatStream = async (body: Record<string, unknown>) => {
    const res = await post("/v1/chat/completions", { ...body, stream: true });
    expect(res.status, res.text.slice(0, 500)).toBe(200);
    const events = sseEvents(res.text);
    const deltas = events.map((e) => e.choices?.[0]?.delta ?? {});
    return {
      raw: res.text,
      events,
      roles: deltas.map((d) => d.role).filter(Boolean),
      content: deltas.map((d) => d.content ?? "").join(""),
      reasoning: deltas.map((d) => d.reasoning_content ?? "").join(""),
      images: deltas.flatMap((d) => d.images ?? []),
      finish: events.map((e) => e.choices?.[0]?.finish_reason).filter(Boolean),
    };
  };
  const ready = (ctx: { skip: () => void }) => {
    if (!app) ctx.skip();
    return app !== undefined;
  };

  beforeAll(async () => {
    const etcd = new EtcdClient();
    if (!(await etcd.ping())) return;

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
        res.statusCode = 200;
        if (r.path.endsWith(":streamGenerateContent")) {
          res.setHeader("content-type", "text/event-stream");
          res.end(framesOf(parts, usageMetadata, finishReason).map((f) => `data: ${JSON.stringify(f)}\r\n\r\n`).join(""));
          return;
        }
        res.setHeader("content-type", "application/json");
        res.end(JSON.stringify(reply(parts, usageMetadata, finishReason)));
      });
    });
    await new Promise<void>((resolve) => upstream!.listen(0, "127.0.0.1", resolve));
    const addr = upstream.address();
    if (addr === null || typeof addr === "string") throw new Error("no port");

    const openRouterImage = { type: "image_url", image_url: { url: `data:image/png;base64,${IMAGE_A}` }, index: 3 };
    openAiCompatible = await startOpenAiUpstream({
      nonStreamBody: {
        id: "or-1",
        object: "chat.completion",
        created: 1_700_000_000,
        model: "or-model",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "from upstream", images: [openRouterImage] },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 5, completion_tokens: 3, total_tokens: 8 },
      },
    });
    openAiCompatibleStream = await startOpenAiUpstream({
      streamEvents: [
        JSON.stringify({
          id: "or-1",
          object: "chat.completion.chunk",
          model: "or-model",
          choices: [{ index: 0, delta: { role: "assistant", images: [openRouterImage] } }],
        }),
        JSON.stringify({
          id: "or-1",
          object: "chat.completion.chunk",
          model: "or-model",
          choices: [{ index: 0, delta: { content: "from upstream" }, finish_reason: "stop" }],
        }),
        "[DONE]",
      ],
    });
    sls = await startMockSls();
    otlp = await startMockOtlp();

    app = await spawnApp({
      exposeDebugHeaders: true,
      logLevel: "debug",
      extraEnv: {
        [`SLS_CRED_${CAPTURE_CREDENTIAL.toUpperCase()}_AK_ID`]: "mock-akid",
        [`SLS_CRED_${CAPTURE_CREDENTIAL.toUpperCase()}_AK_SECRET`]: "mock-secret",
      },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    const pk = await seed.createProviderKey({
      display_name: "vertex-images-pk",
      provider: "google",
      adapter: "vertex",
      secret: JSON.stringify({ access_token: "ya29.images", project: "proj-images", region: "us-central1" }),
      api_base: `http://127.0.0.1:${addr.port}`,
    });
    const model = (display_name: string) =>
      seed.createModel({ display_name, provider: "google", model_name: "gemini-image-e2e", provider_key_id: pk.id });
    await model("img");
    const blocked = await model("img-keyword");
    const masked = await model("img-pii");
    const tight = await model("img-tight-cap");
    const custom = await model("img-custom-block");
    for (const [display_name, compatible] of [
      ["or-img", openAiCompatible],
      ["or-img-stream", openAiCompatibleStream],
    ] as const) {
      const orPk = await seed.createProviderKey({
        display_name: `${display_name}-pk`,
        secret: "sk-mock",
        api_base: `${compatible.baseUrl}/v1`,
      });
      await seed.createModel({
        display_name,
        provider: "openai",
        model_name: "or-model",
        provider_key_id: orPk.id,
      });
    }

    const scoped = async (modelId: string, row: Record<string, unknown>) => {
      const guardrail = await seed.createGuardrail(row, { attach: false });
      await seed.attachGuardrailToModel(guardrail.id, modelId);
    };
    await scoped(blocked.id, {
      name: "img-keyword-output",
      enabled: true,
      hook_point: "output",
      kind: "keyword",
      patterns: [{ kind: "literal", value: BLOCKED_WORD }],
    });
    await scoped(masked.id, {
      name: "img-pii-output",
      enabled: true,
      hook_point: "output",
      kind: "pii",
      detectors: [{ type: "email", action: "mask" }],
    });
    await scoped(custom.id, {
      name: "img-custom-block",
      enabled: true,
      hook_point: "output",
      kind: "custom",
      output_fail_open: false,
      timeout_ms: 5000,
      script: `export function checkOutput() { return { action: "block" }; }`,
    });
    // Its raw-byte bound is far below the image the case below sends.
    await scoped(tight.id, {
      name: "img-pii-tight",
      enabled: true,
      hook_point: "output",
      kind: "pii",
      detectors: [{ type: "email", action: "mask" }],
      max_buffer_bytes: 1024,
      on_buffer_exceeded: "fail_closed",
    });

    await seed.createCachePolicy({ name: "img-cache", enabled: true, applies_to: "all" });
    await seed.createObservabilityExporter({
      name: "img-sls-full",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: "aisix-e2e-images",
      logstore: LOGSTORE,
      credential_ref: CAPTURE_CREDENTIAL,
      content_mode: "full",
    });
    await seed.createObservabilityExporter({
      name: "img-otlp-full",
      enabled: true,
      kind: "otlp_http",
      endpoint: otlp.url,
      content_mode: "full",
    });
    await seed.createApiKey({ key_hash: CALLER_HASH, allowed_models: ["*"] });
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
        method: "POST",
        headers: HEADERS,
        body: JSON.stringify({ model: "or-img", messages: say("warm-up") }),
      });
      await res.text();
      return res.status === 200;
    });
    await waitConfigPropagation(async () => {
      useAnswer([{ text: BLOCKED_WORD }]);
      const res = await post("/v1/chat/completions", { model: "img-keyword", messages: say() });
      useAnswer(FULL_PARTS);
      return res.status === 422;
    });
  }, 60_000);

  afterAll(async () => {
    await app?.exit();
    await new Promise<void>((resolve) => (upstream ? upstream.close(() => resolve()) : resolve()));
    await openAiCompatible?.close();
    await openAiCompatibleStream?.close();
    await sls?.close();
    await otlp?.close();
  });

  test("a whole response carries thought and final images in order, never in content", async (ctx) => {
    if (!ready(ctx)) return;
    useAnswer(FULL_PARTS);
    const message = await chat({ model: "img", messages: say() });
    expect(message.content).toBe("Here it is.");
    expect(message.reasoning_content).toBe(THOUGHT_TEXT.text + MORE_THOUGHT.text);
    expect(message.images).toEqual(FULL_IMAGES);
    expect(JSON.stringify(message)).not.toContain("QVVESU8=");
  });

  test("a stream numbers its images across frames and leaves them out of content", async (ctx) => {
    if (!ready(ctx)) return;
    useAnswer(FULL_PARTS);
    const out = await chatStream({ model: "img", messages: say() });
    expect(out.roles).toEqual(["assistant"]);
    expect(out.content).toBe("Here it is.");
    expect(out.reasoning).toBe(THOUGHT_TEXT.text + MORE_THOUGHT.text);
    expect(out.images).toEqual(FULL_IMAGES);
    expect(out.finish).toEqual(["stop"]);
    expect(out.raw).not.toContain("QVVESU8=");
  });

  test("an image-only answer arrives with empty content", async (ctx) => {
    if (!ready(ctx)) return;
    useAnswer([inline(IMAGE_A)]);
    const message = await chat({ model: "img", messages: say() });
    expect(message.content ?? "").toBe("");
    expect(message.images).toEqual([entry(IMAGE_A, 0)]);

    const out = await chatStream({ model: "img", messages: say() });
    expect(out.roles).toEqual(["assistant"]);
    expect(out.content).toBe("");
    expect(out.images).toEqual([entry(IMAGE_A, 0)]);
    expect(out.finish).toEqual(["stop"]);
  });

  test("a thought image alone is returned like any other", async (ctx) => {
    if (!ready(ctx)) return;
    useAnswer([THOUGHT_TEXT, { ...inline(THOUGHT_IMAGE), thought: true }, { text: "a sketch" }]);
    const message = await chat({ model: "img", messages: say() });
    expect(message.content).toBe("a sketch");
    expect(message.reasoning_content).toBe(THOUGHT_TEXT.text);
    expect(message.images).toEqual([entry(THOUGHT_IMAGE, 0)]);

    const out = await chatStream({ model: "img", messages: say() });
    expect(out.content).toBe("a sketch");
    expect(out.images).toEqual([entry(THOUGHT_IMAGE, 0)]);
  });

  test.for([
    "IMAGE_SAFETY",
    "IMAGE_PROHIBITED_CONTENT",
    "IMAGE_RECITATION",
    "IMAGE_OTHER",
    "NO_IMAGE",
  ])("%s with no parts finishes as content_filter", async (reason, ctx) => {
    if (!ready(ctx)) return;
    useAnswer([], USAGE_METADATA, reason);
    const message = await post("/v1/chat/completions", { model: "img", messages: say() });
    expect(message.status, message.text.slice(0, 500)).toBe(200);
    const choice = JSON.parse(message.text).choices[0];
    expect(choice.finish_reason).toBe("content_filter");
    expect(choice.message).not.toHaveProperty("images");

    const out = await chatStream({ model: "img", messages: say() });
    expect(out.raw).not.toContain("event: error");
    expect(out.roles).toEqual(["assistant"]);
    expect(out.finish).toEqual(["content_filter"]);
    expect(out.images).toEqual([]);
  });

  test("a refused image reaches the other two protocols as a refusal", async (ctx) => {
    if (!ready(ctx)) return;
    useAnswer([], USAGE_METADATA, "IMAGE_RECITATION");
    const messages = await post("/v1/messages", { model: "img", max_tokens: 100, messages: HI });
    expect(messages.status, messages.text.slice(0, 500)).toBe(200);
    expect(JSON.parse(messages.text).stop_reason).toBe("refusal");
    const messagesStream = await post("/v1/messages", { model: "img", max_tokens: 100, messages: HI, stream: true });
    expect(messagesStream.status).toBe(200);
    expect(messagesStream.text).not.toContain("event: error");
    expect(messagesStream.text).toContain('"stop_reason":"refusal"');
    expect(messagesStream.text).toContain("message_stop");

    const responses = await post("/v1/responses", { model: "img", input: "hi" });
    expect(responses.status, responses.text.slice(0, 500)).toBe(200);
    const body = JSON.parse(responses.text);
    expect(body.status).toBe("incomplete");
    expect(body.incomplete_details.reason).toBe("content_filter");
    const responsesStream = await post("/v1/responses", { model: "img", input: "hi", stream: true });
    expect(responsesStream.status).toBe(200);
    expect(responsesStream.text).toContain("response.incomplete");
    expect(responsesStream.text).toContain('"reason":"content_filter"');
  });

  test("a multi-megabyte image survives both paths byte for byte", async (ctx) => {
    if (!ready(ctx)) return;
    const big = image("big", 6 * 1024 * 1024);
    useAnswer([{ text: "large" }, inline(big)]);
    const message = await chat({ model: "img", messages: say() });
    expect(message.images[0].image_url.url === `data:image/png;base64,${big}`).toBe(true);

    const out = await chatStream({ model: "img", messages: say() });
    expect(out.content).toBe("large");
    expect(out.images).toHaveLength(1);
    expect(out.images[0].image_url.url === `data:image/png;base64,${big}`).toBe(true);
  }, 60_000);

  test("modalities reach responseModalities upper-cased, and only when sent", async (ctx) => {
    if (!ready(ctx)) return;
    useAnswer([{ text: "ok" }]);
    for (const stream of [false, true]) {
      received.length = 0;
      const first = await post("/v1/chat/completions", {
        model: "img",
        messages: say(),
        stream,
        modalities: ["text", "image"],
      });
      expect(first.status).toBe(200);
      expect(received.at(-1)!.body.generationConfig.responseModalities, `stream=${stream}`).toEqual(["TEXT", "IMAGE"]);

      const second = await post("/v1/chat/completions", { model: "img", messages: say(), stream });
      expect(second.status).toBe(200);
      expect(received.at(-1)!.body.generationConfig?.responseModalities, `stream=${stream}`).toBeUndefined();
    }
  });

  test("assistant images in history go back as inlineData parts, unsigned", async (ctx) => {
    if (!ready(ctx)) return;
    useAnswer([{ text: "ok" }]);
    const images = [entry(IMAGE_A, 0), entry(IMAGE_B, 1, "image/jpeg")];
    const call = { id: "call_1", type: "function", function: { name: "now", arguments: "{}" } };
    const history = [
      { role: "user", content: "draw" },
      { role: "assistant", content: "Here it is.", images },
      { role: "user", content: "again" },
      { role: "assistant", content: "", images: [entry(IMAGE_A, 0)] },
      { role: "user", content: "and again" },
      { role: "assistant", content: "", reasoning_content: "thinking", images: [entry(IMAGE_B, 0, "image/jpeg")] },
      { role: "user", content: "one more" },
      { role: "assistant", content: "", images: [entry(IMAGE_A, 0)], tool_calls: [call] },
      { role: "tool", tool_call_id: "call_1", content: "12:00" },
    ];
    for (const stream of [false, true]) {
      received.length = 0;
      const res = await post("/v1/chat/completions", { model: "img", messages: history, stream, tools: [NOW] });
      expect(res.status, res.text.slice(0, 500)).toBe(200);
      const turns = received.at(-1)!.body.contents.filter((c: any) => c.role === "model");
      const a = inline(IMAGE_A);
      const b = inline(IMAGE_B, "image/jpeg");
      expect(turns[0].parts, `stream=${stream}`).toEqual([{ text: "Here it is." }, a, b]);
      expect(turns[1].parts, `stream=${stream}`).toEqual([a]);
      expect(turns[2].parts, `stream=${stream}`).toEqual([b]);
      expect(turns[3].parts[0], `stream=${stream}`).toEqual(a);
      expect(turns[3].parts[1].functionCall.name, `stream=${stream}`).toBe("now");
    }
  });

  test("an assistant image that cannot be written is refused naming it", async (ctx) => {
    if (!ready(ctx)) return;
    const before = received.length;
    const res = await post("/v1/chat/completions", {
      model: "img",
      messages: [
        { role: "user", content: "draw" },
        { role: "assistant", content: "x", images: [{ type: "image_url", image_url: {} }] },
        { role: "user", content: "again" },
      ],
    });
    expect(res.status).toBe(400);
    expect(JSON.parse(res.text).error.message).toContain("messages[1].images[0]");

    // Media downloads cover user turns; an http URL here is not fetched.
    const unfetched = await post("/v1/chat/completions", {
      model: "img",
      messages: [
        { role: "user", content: "draw" },
        { role: "assistant", content: "x", images: [{ type: "image_url", image_url: { url: "http://example.com/a.png" } }] },
        { role: "user", content: "again" },
      ],
    });
    expect(unfetched.status).toBe(400);
    expect(JSON.parse(unfetched.text).error.message).toContain("messages[1].images[0]");

    const notAnArray = await post("/v1/chat/completions", {
      model: "img",
      messages: [
        { role: "user", content: "draw" },
        { role: "assistant", content: "x", images: "oops" },
        { role: "user", content: "again" },
      ],
    });
    expect(notAnArray.status).toBe(400);
    expect(JSON.parse(notAnArray.text).error.message).toContain("messages[1].images");
    expect(received.length).toBe(before);
  });

  describe("output guardrails", () => {
    test("do not stand between an image-only answer and the client", async (ctx) => {
      if (!ready(ctx)) return;
      useAnswer([inline(IMAGE_A)]);
      for (const model of ["img-keyword", "img-pii"]) {
        const message = await chat({ model, messages: say() });
        expect(message.images, model).toEqual([entry(IMAGE_A, 0)]);
        const out = await chatStream({ model, messages: say() });
        expect(out.raw, model).not.toContain("event: error");
        expect(out.images, model).toEqual([entry(IMAGE_A, 0)]);
      }
    });

    test("a script that blocks unconditionally decides an image-only answer", async (ctx) => {
      if (!ready(ctx)) return;
      useAnswer([inline(IMAGE_A)]);
      const whole = await post("/v1/chat/completions", { model: "img-custom-block", messages: say() });
      expect(whole.status).toBe(422);
      expect(whole.text).not.toContain(MARKER);
      for (const [path, body] of [
        ["/v1/chat/completions", { messages: say() }],
        ["/v1/messages", { max_tokens: 100, messages: say() }],
        ["/v1/responses", { input: "hi" }],
      ] as const) {
        const res = await post(path, { model: "img-custom-block", ...body, stream: true });
        expect(res.text, path).toContain("blocked by content policy");
        expect(res.text, path).not.toContain(MARKER);
      }
    });

    test("do not read the image: a blocked word inside its bytes passes", async (ctx) => {
      if (!ready(ctx)) return;
      const data = image(`${BLOCKED_WORD}`);
      useAnswer([{ text: "clean" }, inline(data)]);
      const message = await chat({ model: "img-keyword", messages: say() });
      expect(message.images).toEqual([entry(data, 0)]);
      const out = await chatStream({ model: "img-keyword", messages: say() });
      expect(out.raw).not.toContain("event: error");
      expect(out.images).toEqual([entry(data, 0)]);
    });

    test("still block on the text beside an image", async (ctx) => {
      if (!ready(ctx)) return;
      useAnswer([{ text: `has ${BLOCKED_WORD}` }, inline(IMAGE_A)]);
      const whole = await post("/v1/chat/completions", { model: "img-keyword", messages: say() });
      expect(whole.status).toBe(422);
      expect(whole.text).not.toContain(MARKER);

      const res = await post("/v1/chat/completions", { model: "img-keyword", messages: say(), stream: true });
      expect(res.text).toContain("event: error");
      expect(res.text).not.toContain(BLOCKED_WORD);
      expect(res.text).not.toContain(MARKER);
    });

    test("still mask the text beside an image and leave the image alone", async (ctx) => {
      if (!ready(ctx)) return;
      useAnswer([{ text: `mail ${MASKED_EMAIL}` }, inline(IMAGE_A)]);
      const message = await chat({ model: "img-pii", messages: say() });
      expect(message.content).not.toContain(MASKED_EMAIL);
      expect(message.content).toContain("mail ");
      expect(message.images).toEqual([entry(IMAGE_A, 0)]);

      const out = await chatStream({ model: "img-pii", messages: say() });
      expect(out.content).not.toContain(MASKED_EMAIL);
      expect(out.content).toContain("mail ");
      expect(out.images).toEqual([entry(IMAGE_A, 0)]);
    });

    test("do not trip the hold-back bound on an image's size", async (ctx) => {
      if (!ready(ctx)) return;
      // 128 x max_buffer_bytes is what a held stream may keep in raw bytes.
      const large = image("capped", 1024 * 1024);
      useAnswer([{ text: "short" }, inline(large)]);
      const out = await chatStream({ model: "img-tight-cap", messages: say() });
      expect(out.raw.includes("event: error")).toBe(false);
      expect(out.content).toBe("short");
      expect(out.images).toHaveLength(1);
    });
  });

  describe("content capture", () => {
    const capturedInput = (log: Map<string, string>) => log.get("input_messages") ?? log.get("prompt") ?? "";
    const capturedRow = (marker: string) =>
      waitForSlsLog(sls!, LOGSTORE, (l) => capturedInput(l).includes(marker), marker);

    test("no exporter, log or event receives image data", async (ctx) => {
      if (!ready(ctx)) return;
      const tag = `capture-${++unique}`;
      const answer = `capture-answer-${unique}`;
      const from = sls!.requests.length;
      // A prompt over the capture cap is exported as the raw request JSON, cut
      // down string by string. An image this small survives that cut, so only
      // omitting it from the captured request keeps it from the exporter.
      const padding = "x".repeat(256 * 1024);
      const history = (name: string) => [
        { role: "user", content: "draw" },
        { role: "assistant", content: "before", images: [entry(IMAGE_B, 0)] },
        { role: "user", content: `${tag}-${name} ${padding}` },
      ];
      const rows = (name: string) =>
        slsLogsFor(sls!, LOGSTORE, from).filter((l) => (l.get("prompt") ?? "").includes(`${tag}-${name}`));

      useAnswer([{ text: answer }, inline(IMAGE_A)]);
      await chat({ model: "img", messages: history("whole") });
      await chatStream({ model: "img", messages: history("stream") });

      useAnswer([{ text: answer }]);
      await chat({ model: "img", messages: history("hit") });
      const hit = await post("/v1/chat/completions", { model: "img", messages: history("hit") });
      expect(hit.headers.get("x-aisix-cache")).toBe("hit");

      const failed = await post("/v1/chat/completions", {
        model: "img",
        messages: history("failed"),
        tools: [{ type: "web_search" }],
      });
      expect(failed.status).toBe(400);

      for (const [name, count] of [["whole", 1], ["stream", 1], ["hit", 2], ["failed", 1]] as const) {
        await expect.poll(() => rows(name).length, { timeout: 10_000, message: name }).toBeGreaterThanOrEqual(count);
        for (const row of rows(name)) {
          expect(row.get("prompt"), name).toContain("[aisix: base64 data uri omitted,");
        }
      }
      const captured = decodedTextFor(sls!, LOGSTORE, from);
      expect(captured).toContain(answer);
      expect(captured.includes(MARKER)).toBe(false);

      await expect.poll(() => JSON.stringify(otlp!.spans).includes(`${tag}-failed`), { timeout: 10_000 }).toBe(true);
      expect(JSON.stringify(otlp!.spans).includes(MARKER)).toBe(false);
      expect(app!.output().includes(MARKER)).toBe(false);
    });

    test("with no upstream usage, the estimate counts text only", async (ctx) => {
      if (!ready(ctx)) return;
      const prompt = `estimate-prompt-${++unique}`;
      useAnswer([{ text: "short answer" }, inline(image("estimate", 256 * 1024))], null);
      const history = (last: string) => [
        { role: "user", content: "draw" },
        { role: "assistant", content: "x", images: [entry(image("estimatein", 256 * 1024), 0)] },
        { role: "user", content: last },
      ];
      await chat({ model: "img", messages: history(prompt) });
      await chatStream({ model: "img", messages: history(`${prompt}-s`) });
      for (const marker of [prompt, `${prompt}-s`]) {
        const log = await capturedRow(marker);
        expect(log.get("usage_estimated"), marker).toBe("true");
        expect(Number(log.get("completion_tokens")), marker).toBeGreaterThan(0);
        expect(Number(log.get("completion_tokens")), marker).toBeLessThan(50);
        expect(Number(log.get("prompt_tokens")), marker).toBeGreaterThan(0);
        expect(Number(log.get("prompt_tokens")), marker).toBeLessThan(200);
      }
    });
  });

  test("a response with images is not cached, one with text alone is", async (ctx) => {
    if (!ready(ctx)) return;
    const cachedAcross = async (messages: unknown[]) => {
      const first = await post("/v1/chat/completions", { model: "img", messages });
      expect(first.status).toBe(200);
      const second = await post("/v1/chat/completions", { model: "img", messages });
      expect(second.status).toBe(200);
      return second.headers.get("x-aisix-cache");
    };
    useAnswer([{ text: "plain" }]);
    expect(await cachedAcross(say())).toBe("hit");
    useAnswer([{ text: "with a picture" }, inline(IMAGE_A)]);
    expect(await cachedAcross(say())).toBe("miss");
  });

  test.for([
    ["/v1/messages", { max_tokens: 100, messages: HI }, "hi from text"],
    ["/v1/responses", { input: "hi" }, "hi from text"],
  ] as const)("%s keeps the text and discards the images", async ([path, request, answer], ctx) => {
    if (!ready(ctx)) return;
    for (const answerParts of [[{ text: answer }, inline(IMAGE_A), inline(IMAGE_B)], [inline(IMAGE_A)]]) {
      useAnswer(answerParts);
      const texted = answerParts.length > 1;
      const whole = await post(path, { model: "img", ...request });
      expect(whole.status, whole.text.slice(0, 500)).toBe(200);
      expect(whole.text).not.toContain(MARKER);
      expect(whole.text.includes(answer)).toBe(texted);

      const streamed = await post(path, { model: "img", ...request, stream: true });
      expect(streamed.status, streamed.text.slice(0, 500)).toBe(200);
      expect(streamed.text).not.toContain(MARKER);
      expect(streamed.text).not.toContain("event: error");
      expect(streamed.text.includes(answer)).toBe(texted);
      // The stream ran to its protocol's own end.
      expect(streamed.text).toContain(path === "/v1/messages" ? "message_stop" : "response.completed");
    }

    // A held stream does not trip its bound on an image it was going to discard.
    useAnswer([{ text: answer }, inline(image("discarded", 1024 * 1024))]);
    const held = await post(path, { model: "img-tight-cap", ...request, stream: true });
    expect(held.status, held.text.slice(0, 500)).toBe(200);
    expect(held.text).not.toContain("event: error");
    expect(held.text).not.toContain(MARKER);
    expect(held.text).toContain(answer);
    expect(held.text).toContain(path === "/v1/messages" ? "message_stop" : "response.completed");
  });

  test("images from an OpenAI-compatible upstream pass through chat unchanged", async (ctx) => {
    if (!ready(ctx)) return;
    const expected = { type: "image_url", image_url: { url: `data:image/png;base64,${IMAGE_A}` }, index: 3 };
    const message = await chat({ model: "or-img", messages: say() });
    expect(message.content).toBe("from upstream");
    expect(message.images).toEqual([expected]);

    const out = await chatStream({ model: "or-img-stream", messages: say() });
    expect(out.content).toBe("from upstream");
    expect(out.images).toEqual([expected]);
  });
});
