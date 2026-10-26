import { createHash } from "node:crypto";
import { createServer, type Server } from "node:http";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// Request parameters are handled by what the target platform's contract
// defines: a field the target takes is written as the caller sent it (or
// into the slot of the same meaning), a field it cannot represent is
// dropped, a value illegal in the caller's own protocol is refused with a
// 400 naming it, and whatever the upstream answers comes back in full.

const CALLER_PLAINTEXT = "sk-param-contract";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");

const HEADERS = {
  authorization: `Bearer ${CALLER_PLAINTEXT}`,
  "content-type": "application/json",
};

const LOGPROBS = {
  content: [{ token: "Hi", logprob: -0.25, bytes: [72, 105], top_logprobs: [] }],
  refusal: null,
};

// Two choices: the first with logprobs and a web-search annotation, the
// second a refusal next to a legacy function call.
const MULTI_CHOICE_REPLY = {
  id: "chatcmpl-multi",
  object: "chat.completion",
  created: 1_700_000_000,
  model: "gpt-4o",
  choices: [
    {
      index: 0,
      message: {
        role: "assistant",
        content: "Hi",
        refusal: null,
        annotations: [
          { type: "url_citation", url_citation: { url: "https://example.com", title: "ex", start_index: 0, end_index: 2 } },
        ],
      },
      logprobs: LOGPROBS,
      finish_reason: "stop",
    },
    {
      index: 1,
      message: {
        role: "assistant",
        content: null,
        refusal: "I can't help with that.",
        function_call: { name: "lookup", arguments: "{\"q\":\"x\"}" },
      },
      logprobs: null,
      finish_reason: "function_call",
    },
  ],
  usage: { prompt_tokens: 9, completion_tokens: 6, total_tokens: 15 },
};

function chunk(choices: unknown[]): string {
  return JSON.stringify({
    id: "chatcmpl-stream",
    object: "chat.completion.chunk",
    created: 1_700_000_000,
    model: "gpt-4o",
    choices,
  });
}

// Two interleaved choices: the first streams a legacy function call with
// logprobs, the second a refusal.
const MULTI_CHOICE_STREAM = [
  chunk([{ index: 0, delta: { role: "assistant", function_call: { name: "lookup", arguments: "" } }, logprobs: LOGPROBS, finish_reason: null }]),
  chunk([{ index: 1, delta: { role: "assistant", refusal: "I can't" }, logprobs: null, finish_reason: null }]),
  chunk([{ index: 0, delta: { function_call: { arguments: "{\"q\":\"x\"}" } }, logprobs: null, finish_reason: null }]),
  chunk([
    { index: 0, delta: {}, logprobs: null, finish_reason: "function_call" },
    { index: 1, delta: {}, logprobs: null, finish_reason: "stop" },
  ]),
  "[DONE]",
];

const ARRAY_CONTENT_REPLY = {
  id: "chatcmpl-array",
  object: "chat.completion",
  created: 1_700_000_000,
  model: "magistral",
  choices: [
    {
      index: 0,
      message: { role: "assistant", content: [{ type: "text", text: "from parts" }] },
      finish_reason: "stop",
    },
  ],
  usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 },
};

const PLAIN_REPLY = {
  id: "chatcmpl-plain",
  object: "chat.completion",
  created: 1_700_000_000,
  model: "gpt-4o",
  choices: [{ index: 0, message: { role: "assistant", content: "backup" }, finish_reason: "stop" }],
  usage: { prompt_tokens: 3, completion_tokens: 1, total_tokens: 4 },
};

const UPSTREAM_ERROR_MESSAGE = "The model `ghost` does not exist or you do not have access to it.";

const EMBEDDING_REPLY = {
  object: "list",
  model: "text-embedding-3-small",
  data: [{ object: "embedding", index: 0, embedding: [0.1, 0.2] }],
  usage: { prompt_tokens: 3, total_tokens: 3 },
};

const WEATHER_FUNCTION = {
  name: "get_weather",
  description: "Current weather for a city",
  parameters: {
    type: "object",
    properties: { city: { type: "string" } },
    required: ["city"],
  },
};

// A legacy function-calling conversation: the caller's earlier call and
// its `function` result, then a new question.
const LEGACY_REQUEST = {
  messages: [
    { role: "user", content: "Weather in Paris?" },
    { role: "assistant", content: null, function_call: { name: "get_weather", arguments: "{\"city\":\"Paris\"}" } },
    { role: "function", name: "get_weather", content: "sunny" },
    { role: "user", content: "And in Rome?" },
  ],
  functions: [WEATHER_FUNCTION],
  function_call: "auto",
};

const ANTHROPIC_TOOL_USE_REPLY = {
  id: "msg_legacy",
  type: "message",
  role: "assistant",
  model: "claude-sonnet-4-5",
  content: [{ type: "tool_use", id: "toolu_rome", name: "get_weather", input: { city: "Rome" } }],
  stop_reason: "tool_use",
  usage: { input_tokens: 20, output_tokens: 8 },
};

function sseFrame(event: string, data: unknown): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

const ANTHROPIC_TOOL_USE_STREAM = [
  sseFrame("message_start", {
    type: "message_start",
    message: { id: "msg_s", type: "message", role: "assistant", model: "claude-sonnet-4-5", content: [], stop_reason: null, usage: { input_tokens: 20, output_tokens: 1 } },
  }),
  sseFrame("content_block_start", {
    type: "content_block_start",
    index: 0,
    content_block: { type: "tool_use", id: "toolu_rome", name: "get_weather", input: {} },
  }),
  sseFrame("content_block_delta", {
    type: "content_block_delta",
    index: 0,
    delta: { type: "input_json_delta", partial_json: "{\"city\":" },
  }),
  sseFrame("content_block_delta", {
    type: "content_block_delta",
    index: 0,
    delta: { type: "input_json_delta", partial_json: "\"Rome\"}" },
  }),
  sseFrame("content_block_stop", { type: "content_block_stop", index: 0 }),
  sseFrame("message_delta", { type: "message_delta", delta: { stop_reason: "tool_use" }, usage: { output_tokens: 8 } }),
  sseFrame("message_stop", { type: "message_stop" }),
];

const GEMINI_CALL_REPLY = {
  candidates: [
    {
      content: { role: "model", parts: [{ functionCall: { name: "get_weather", args: { city: "Rome" } } }] },
      finishReason: "STOP",
    },
  ],
  usageMetadata: { promptTokenCount: 20, candidatesTokenCount: 5, totalTokenCount: 25 },
};

function post(app: SpawnedApp, path: string, body: unknown): Promise<Response> {
  return fetch(`${app.proxyUrl}${path}`, { method: "POST", headers: HEADERS, body: JSON.stringify(body) });
}

/** The JSON payloads of every `data:` line of an SSE body. */
function sseData(raw: string): any[] {
  return raw
    .split(/\r?\n/)
    .filter((l) => l.startsWith("data: ") && !l.includes("[DONE]"))
    .map((l) => JSON.parse(l.slice(6)));
}

/** The body of the last request `upstream` received after `baseline`. */
function lastBody(upstream: OpenAiUpstream, baseline: number): { raw: string; json: any } {
  const calls = upstream.receivedRequests.slice(baseline);
  expect(calls.length, "the upstream was called").toBeGreaterThan(0);
  const raw = calls.at(-1)!.body;
  return { raw, json: JSON.parse(raw) };
}

describe("request parameters follow the target platform's contract", () => {
  let app: SpawnedApp | undefined;
  let etcdReachable = false;
  const up: Record<string, OpenAiUpstream> = {};

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    up.multi = await startOpenAiUpstream({ nonStreamBody: MULTI_CHOICE_REPLY });
    up.stream = await startOpenAiUpstream({ streamEvents: MULTI_CHOICE_STREAM });
    up.garbage = await startOpenAiUpstream({ rawBody: "<html>not json</html>", rawContentType: "text/html" });
    up.backup = await startOpenAiUpstream({ nonStreamBody: PLAIN_REPLY });
    up.array = await startOpenAiUpstream({ nonStreamBody: ARRAY_CONTENT_REPLY });
    up.error = await startOpenAiUpstream({
      status: 404,
      errorBody: { error: { message: UPSTREAM_ERROR_MESSAGE, type: "invalid_request_error", code: 404, param: null } },
    });
    up.embed = await startOpenAiUpstream({ nonStreamBody: EMBEDDING_REPLY });
    up.claude = await startOpenAiUpstream({ nonStreamBody: ANTHROPIC_TOOL_USE_REPLY });
    up.claudeStream = await startOpenAiUpstream({ rawStreamFrames: ANTHROPIC_TOOL_USE_STREAM });
    up.gemini = await startOpenAiUpstream({ nonStreamBody: GEMINI_CALL_REPLY });

    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);
    const openaiModel = async (name: string, upstream: OpenAiUpstream, modelName = "gpt-4o") => {
      const pk = await seed.createProviderKey({
        display_name: `${name}-pk`,
        secret: "sk-mock",
        api_base: `${upstream.baseUrl}/v1`,
      });
      await seed.createModel({ display_name: name, provider: "openai", model_name: modelName, provider_key_id: pk.id });
    };
    await openaiModel("pc-multi", up.multi);
    await openaiModel("pc-stream", up.stream);
    await openaiModel("pc-garbage", up.garbage);
    await openaiModel("pc-backup", up.backup);
    await openaiModel("pc-array", up.array);
    await openaiModel("pc-error", up.error);
    await openaiModel("pc-embed", up.embed, "text-embedding-3-small");
    await seed.createModel({
      display_name: "pc-group",
      routing: { strategy: "failover", targets: [{ model: "pc-garbage" }, { model: "pc-backup" }] },
    });

    for (const [name, upstream] of [
      ["pc-claude", up.claude],
      ["pc-claude-stream", up.claudeStream],
    ] as const) {
      const pk = await seed.createProviderKey({
        display_name: `${name}-pk`,
        provider: "anthropic",
        adapter: "anthropic",
        secret: "sk-ant-mock",
        api_base: upstream.baseUrl,
      });
      await seed.createModel({
        display_name: name,
        provider: "anthropic",
        model_name: "claude-sonnet-4-5",
        provider_key_id: pk.id,
      });
    }

    const vertexPk = await seed.createProviderKey({
      display_name: "pc-gemini-pk",
      provider: "google",
      adapter: "vertex",
      secret: JSON.stringify({ access_token: "ya29.param-contract", project: "proj-e2e", region: "us-central1" }),
      api_base: up.gemini.baseUrl,
    });
    await seed.createModel({
      display_name: "pc-gemini",
      provider: "google",
      model_name: "gemini-2.5-flash",
      provider_key_id: vertexPk.id,
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
    await Promise.all(Object.values(up).map((u) => u.close()));
  });

  // ── answers come back in full ──────────────────────────────────────

  test("every choice, its logprobs, refusal, annotations and legacy function_call come back", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const res = await post(app, "/v1/chat/completions", {
      model: "pc-multi",
      messages: [{ role: "user", content: "hi" }],
      n: 2,
      logprobs: true,
    });
    const body = (await res.json()) as any;
    expect(res.status, JSON.stringify(body)).toBe(200);
    expect(body.choices).toHaveLength(2);
    expect(body.choices[0].logprobs).toEqual(LOGPROBS);
    expect(body.choices[0].message.annotations).toEqual(MULTI_CHOICE_REPLY.choices[0].message.annotations);
    expect(body.choices[1].index).toBe(1);
    expect(body.choices[1].message.refusal).toBe("I can't help with that.");
    expect(body.choices[1].message.function_call).toEqual({ name: "lookup", arguments: "{\"q\":\"x\"}" });
    expect(body.choices[1].finish_reason).toBe("function_call");
  });

  test("a streamed answer keeps every choice, keyed by its index", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const res = await post(app, "/v1/chat/completions", {
      model: "pc-stream",
      messages: [{ role: "user", content: "hi" }],
      n: 2,
      stream: true,
    });
    const text = await res.text();
    expect(res.status, text).toBe(200);
    const choices = sseData(text).flatMap((c) => c.choices ?? []);
    const of = (i: number) => choices.filter((c) => c.index === i);
    expect(of(1).map((c) => c.delta?.refusal).join("")).toBe("I can't");
    expect(of(1).at(-1)?.finish_reason).toBe("stop");
    expect(of(0).find((c) => c.logprobs)?.logprobs).toEqual(LOGPROBS);
    expect(of(0).find((c) => c.delta?.function_call?.name)?.delta.function_call.name).toBe("lookup");
    expect(of(0).map((c) => c.delta?.function_call?.arguments ?? "").join("")).toBe("{\"q\":\"x\"}");
    expect(of(0).at(-1)?.finish_reason).toBe("function_call");
  });

  test("an undecodable 2xx is answered 502 and not run again on the next target", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const garbageBefore = up.garbage.receivedRequests.length;
    const backupBefore = up.backup.receivedRequests.length;
    const res = await post(app, "/v1/chat/completions", {
      model: "pc-group",
      messages: [{ role: "user", content: "hi" }],
    });
    const text = await res.text();
    expect(res.status, text).toBe(502);
    expect(up.garbage.receivedRequests.length - garbageBefore).toBe(1);
    expect(up.backup.receivedRequests.length - backupBefore).toBe(0);
  });

  test("an answer whose content is an array of parts is relayed", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const res = await post(app, "/v1/chat/completions", {
      model: "pc-array",
      messages: [{ role: "user", content: "hi" }],
    });
    const body = (await res.json()) as any;
    expect(res.status, JSON.stringify(body)).toBe(200);
    expect(body.choices[0].message.content).toEqual([{ type: "text", text: "from parts" }]);
  });

  test("an upstream error with a numeric code keeps the upstream's message", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const res = await post(app, "/v1/chat/completions", {
      model: "pc-error",
      messages: [{ role: "user", content: "hi" }],
    });
    const body = (await res.json()) as any;
    expect(res.status, JSON.stringify(body)).toBe(404);
    expect(body.error.message).toBe(UPSTREAM_ERROR_MESSAGE);
  });

  // ── fields the target takes go out as sent ─────────────────────────

  /** Send `extra` to the multi-choice upstream and return the body it received. */
  async function sentToUpstream(extra: Record<string, unknown>): Promise<any> {
    const baseline = up.multi.receivedRequests.length;
    const res = await post(app!, "/v1/chat/completions", {
      model: "pc-multi",
      messages: [{ role: "user", content: "hi" }],
      ...extra,
    });
    const text = await res.text();
    expect(res.status, text).toBe(200);
    return lastBody(up.multi, baseline).json;
  }

  test("a strict schema the caller wrote goes out as written", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const schema = {
      type: "object",
      properties: { a: { type: "string" }, b: { type: "string" } },
      required: ["a"],
    };
    const sent = await sentToUpstream({
      response_format: { type: "json_schema", json_schema: { name: "answer", strict: true, schema } },
    });
    expect(sent.response_format.json_schema.schema).toEqual(schema);
  });

  test("an assistant's null content goes out as null", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const sent = await sentToUpstream({
      messages: [
        { role: "user", content: "hi" },
        {
          role: "assistant",
          content: null,
          tool_calls: [{ id: "call_1", type: "function", function: { name: "f", arguments: "{}" } }],
        },
        { role: "tool", tool_call_id: "call_1", content: "ok" },
      ],
    });
    expect(sent.messages[1].content).toBeNull();
  });

  test("temperature and top_p go out as the decimals the caller wrote", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const sent = await sentToUpstream({ temperature: 0.7, top_p: 0.9 });
    expect(sent.temperature).toBe(0.7);
    expect(sent.top_p).toBe(0.9);
  });

  test("embeddings forward user, unknown fields and token-id input", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const baseline = up.embed.receivedRequests.length;
    const res = await post(app, "/v1/embeddings", {
      model: "pc-embed",
      input: [[101, 2023, 102]],
      user: "user-42",
      x_trace: "t-1",
    });
    const text = await res.text();
    expect(res.status, text).toBe(200);
    const sent = lastBody(up.embed, baseline).json;
    expect(sent.input).toEqual([[101, 2023, 102]]);
    expect(sent.user).toBe("user-42");
    expect(sent.x_trace).toBe("t-1");
  });

  // ── legacy function calling ────────────────────────────────────────

  test("an OpenAI-wire target receives functions, function_call and the function turn as sent", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const baseline = up.multi.receivedRequests.length;
    const res = await post(app, "/v1/chat/completions", { model: "pc-multi", ...LEGACY_REQUEST });
    const body = (await res.json()) as any;
    expect(res.status, JSON.stringify(body)).toBe(200);
    const sent = lastBody(up.multi, baseline).json;
    expect(sent.functions).toEqual([WEATHER_FUNCTION]);
    expect(sent.function_call).toBe("auto");
    expect(sent.messages[1].function_call).toEqual(LEGACY_REQUEST.messages[1].function_call);
    expect(sent.messages[2]).toMatchObject({ role: "function", name: "get_weather", content: "sunny" });
  });

  test("Claude receives legacy function calling as tools and answers in the legacy shape", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const baseline = up.claude.receivedRequests.length;
    const res = await post(app, "/v1/chat/completions", { model: "pc-claude", ...LEGACY_REQUEST });
    const body = (await res.json()) as any;
    expect(res.status, JSON.stringify(body)).toBe(200);

    const sent = lastBody(up.claude, baseline).json;
    expect(sent.functions).toBeUndefined();
    expect(sent.function_call).toBeUndefined();
    expect(sent.tools.map((t: any) => t.name)).toEqual(["get_weather"]);
    expect(sent.tool_choice).toEqual({ type: "auto" });
    const toolUse = sent.messages[1].content.find((b: any) => b.type === "tool_use");
    expect(toolUse).toMatchObject({ name: "get_weather", input: { city: "Paris" } });
    const toolResult = sent.messages[2].content.find((b: any) => b.type === "tool_result");
    expect(toolResult.tool_use_id).toBe(toolUse.id);

    expect(body.choices[0].message.function_call).toEqual({ name: "get_weather", arguments: "{\"city\":\"Rome\"}" });
    expect(body.choices[0].message.tool_calls).toBeUndefined();
    expect(body.choices[0].finish_reason).toBe("function_call");
  });

  test("Claude streams a legacy function call back as delta.function_call", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const res = await post(app, "/v1/chat/completions", { model: "pc-claude-stream", ...LEGACY_REQUEST, stream: true });
    const text = await res.text();
    expect(res.status, text).toBe(200);
    const choices = sseData(text).flatMap((c) => c.choices ?? []);
    expect(choices.some((c) => c.delta?.tool_calls)).toBe(false);
    expect(choices.find((c) => c.delta?.function_call?.name)?.delta.function_call.name).toBe("get_weather");
    expect(choices.map((c) => c.delta?.function_call?.arguments ?? "").join("")).toBe("{\"city\":\"Rome\"}");
    expect(choices.at(-1)?.finish_reason ?? choices.find((c) => c.finish_reason)?.finish_reason).toBe("function_call");
  });

  test("Vertex Gemini receives legacy function calling as function declarations", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const baseline = up.gemini.receivedRequests.length;
    const res = await post(app, "/v1/chat/completions", { model: "pc-gemini", ...LEGACY_REQUEST });
    const body = (await res.json()) as any;
    expect(res.status, JSON.stringify(body)).toBe(200);

    const sent = lastBody(up.gemini, baseline).json;
    expect(sent.tools[0].functionDeclarations.map((f: any) => f.name)).toEqual(["get_weather"]);
    const parts = sent.contents.flatMap((c: any) => c.parts);
    expect(parts.find((p: any) => p.functionCall)?.functionCall).toMatchObject({ name: "get_weather", args: { city: "Paris" } });
    expect(parts.find((p: any) => p.functionResponse)?.functionResponse.name).toBe("get_weather");

    expect(body.choices[0].message.function_call.name).toBe("get_weather");
    expect(JSON.parse(body.choices[0].message.function_call.arguments)).toEqual({ city: "Rome" });
    expect(body.choices[0].finish_reason).toBe("function_call");
  });

  // ── illegal values are refused, naming the field ───────────────────

  test.for([
    { label: "negative max_tokens", patch: { max_tokens: -1 }, field: "max_tokens" },
    { label: "max_tokens past u32", patch: { max_tokens: 2 ** 33 }, field: "max_tokens" },
    { label: "string temperature", patch: { temperature: "hot" }, field: "temperature" },
    { label: "missing messages", patch: { messages: undefined }, field: "messages" },
    { label: "numeric user content", patch: { messages: [{ role: "user", content: 42 }] }, field: "messages[0]" },
    { label: "object system content", patch: { messages: [{ role: "system", content: { text: "x" } }, { role: "user", content: "hi" }] }, field: "messages[0]" },
  ])("$label is a 400 naming the field", async (c, ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const res = await post(app, "/v1/chat/completions", {
      model: "pc-multi",
      messages: [{ role: "user", content: "hi" }],
      ...c.patch,
    });
    const body = (await res.json()) as any;
    expect(res.status, JSON.stringify(body)).toBe(400);
    expect(body.error.type).toBe("invalid_request_error");
    expect(body.error.message).toContain(c.field);
  });
});

/** An OpenAI moderation endpoint that flags any input containing "bomb". */
async function startModeration(): Promise<{ baseUrl: string; close: () => Promise<void> }> {
  const server: Server = createServer((req, res) => {
    let raw = "";
    req.on("data", (c: Buffer) => (raw += c.toString("utf8")));
    req.on("end", () => {
      const input = String(JSON.parse(raw || "{}").input ?? "");
      const flagged = input.includes("bomb");
      res.setHeader("content-type", "application/json");
      res.end(
        JSON.stringify({
          id: "modr",
          model: "omni-moderation-latest",
          results: [{ flagged, categories: { violence: flagged }, category_scores: { violence: flagged ? 0.99 : 0.01 } }],
        }),
      );
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const addr = server.address();
  if (addr === null || typeof addr === "string") throw new Error("no port");
  return {
    baseUrl: `http://127.0.0.1:${addr.port}`,
    close: () => new Promise<void>((resolve, reject) => server.close((e) => (e ? reject(e) : resolve()))),
  };
}

// Output guardrails over the answer as relayed: each choice is scanned on
// its own, and the refusal text is masked like the content.
describe("output guardrails cover every choice and the refusal", () => {
  let app: SpawnedApp | undefined;
  let etcdReachable = false;
  const up: Record<string, OpenAiUpstream> = {};
  const EMAIL = "jane@corp.example";
  let moderationServer: { baseUrl: string; close: () => Promise<void> };

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    // "bomb" split across choice 0's deltas, with choice 1's between them.
    up.interleaved = await startOpenAiUpstream({
      streamEvents: [
        chunk([{ index: 0, delta: { role: "assistant", content: "bo" }, finish_reason: null }]),
        chunk([{ index: 1, delta: { role: "assistant", content: "xx" }, finish_reason: null }]),
        chunk([{ index: 0, delta: { content: "mb" }, finish_reason: null }]),
        chunk([
          { index: 0, delta: {}, finish_reason: "stop" },
          { index: 1, delta: {}, finish_reason: "stop" },
        ]),
        "[DONE]",
      ],
    });
    up.refusal = await startOpenAiUpstream({
      nonStreamBody: {
        id: "chatcmpl-r",
        object: "chat.completion",
        created: 1_700_000_000,
        model: "gpt-4o",
        choices: [{ index: 0, message: { role: "assistant", content: null, refusal: `I can't email ${EMAIL}` }, finish_reason: "stop" }],
        usage: { prompt_tokens: 3, completion_tokens: 4, total_tokens: 7 },
      },
    });
    up.refusalStream = await startOpenAiUpstream({
      streamEvents: [
        chunk([{ index: 0, delta: { role: "assistant", refusal: "I can't email jane@" }, finish_reason: null }]),
        chunk([{ index: 0, delta: { refusal: "corp.example" }, finish_reason: null }]),
        chunk([{ index: 0, delta: {}, finish_reason: "stop" }]),
        "[DONE]",
      ],
    });

    moderationServer = await startModeration();

    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);
    const model = async (name: string, upstream: OpenAiUpstream) => {
      const pk = await seed.createProviderKey({ display_name: `${name}-pk`, secret: "sk-mock", api_base: `${upstream.baseUrl}/v1` });
      return seed.createModel({ display_name: name, provider: "openai", model_name: "gpt-4o", provider_key_id: pk.id });
    };
    const interleaved = await model("pg-interleaved", up.interleaved);
    const refusal = await model("pg-refusal", up.refusal);
    const refusalStream = await model("pg-refusal-stream", up.refusalStream);
    // A remote moderation guardrail judges the text it is sent.
    const moderation = await seed.createGuardrail(
      {
        name: "pg-moderation",
        enabled: true,
        kind: "openai_moderation",
        hook_point: "output",
        api_key: "sk-moderation",
        endpoint: moderationServer.baseUrl,
      },
      { attach: false },
    );
    await seed.attachGuardrailToModel(moderation.id, interleaved.id);
    const pii = await seed.createGuardrail(
      { name: "pg-pii", enabled: true, hook_point: "output", kind: "pii", detectors: [{ type: "email", action: "mask" }] },
      { attach: false },
    );
    await seed.attachGuardrailToModel(pii.id, refusal.id);
    await seed.attachGuardrailToModel(pii.id, refusalStream.id);

    // Seeded last: the key authenticating implies the whole seed set is live.
    await seed.createApiKey({ key_hash: CALLER_KEY_HASH, allowed_models: ["*"] });
    await waitConfigPropagation(async () => {
      const res = await post(app!, "/v1/chat/completions", { model: "pg-refusal", messages: [{ role: "user", content: "hi" }] });
      await res.arrayBuffer();
      return res.status === 200;
    });
  });

  afterAll(async () => {
    await app?.exit();
    await Promise.all(Object.values(up).map((u) => u.close()));
    await moderationServer?.close();
  });

  test("a blocked literal split across one choice's deltas is caught between another choice's", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const res = await post(app, "/v1/chat/completions", {
      model: "pg-interleaved",
      n: 2,
      stream: true,
      messages: [{ role: "user", content: "hi" }],
    });
    const text = await res.text();
    const content = sseData(text)
      .flatMap((c: any) => c.choices ?? [])
      .filter((c: any) => c.index === 0)
      .map((c: any) => c.delta?.content ?? "")
      .join("");
    expect(content).not.toContain("bomb");
    expect(text).toContain("event: error");
  });

  test("the refusal text is masked, streamed or not", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const plain = await post(app, "/v1/chat/completions", { model: "pg-refusal", messages: [{ role: "user", content: "hi" }] });
    const body = (await plain.json()) as any;
    expect(plain.status, JSON.stringify(body)).toBe(200);
    expect(body.choices[0].message.refusal).toContain("I can't email");
    expect(body.choices[0].message.refusal).not.toContain(EMAIL);

    const streamed = await post(app, "/v1/chat/completions", {
      model: "pg-refusal-stream",
      stream: true,
      messages: [{ role: "user", content: "hi" }],
    });
    const text = await streamed.text();
    expect(streamed.status, text).toBe(200);
    const refusal = sseData(text)
      .flatMap((c: any) => c.choices ?? [])
      .map((c: any) => c.delta?.refusal ?? "")
      .join("");
    expect(refusal).toContain("I can't email");
    expect(refusal).not.toContain(EMAIL);
  });
});
