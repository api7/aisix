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

// Function calling on a Vertex model served by Gemini's native
// `:generateContent` / `:streamGenerateContent` wire. The caller's
// tools have to become Gemini function declarations, the tool choice a
// function-calling mode, and the `functionCall` parts Gemini answers
// with have to reach every client protocol as that protocol's tool
// call — and the history the client sends back has to reach Gemini as
// `functionCall` / `functionResponse` parts, with the thought signature
// Gemini attached to the call.

const CALLER_PLAINTEXT = "sk-vertex-gemini-function-calling";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

// Gemini signatures are standard base64 and hundreds of bytes long; their
// length, `+`, `/` and padding are what an id must not carry to other
// providers.
const SIGNATURE = "c2ln+bmF0/dXJl".repeat(20) + "c2lnbg==";
const FINAL_ANSWER = "Sunny in Paris, 21 degrees in Rome.";

const WEATHER_TOOL = {
  type: "function",
  function: {
    name: "get_weather",
    description: "Current weather for a city",
    parameters: {
      type: "object",
      properties: { city: { type: "string" } },
      required: ["city"],
      additionalProperties: false,
    },
  },
};

const ANTHROPIC_WEATHER_TOOL = {
  name: "get_weather",
  description: "Current weather for a city",
  input_schema: WEATHER_TOOL.function.parameters,
};

const RESPONSES_WEATHER_TOOL = {
  type: "function",
  name: "get_weather",
  description: "Current weather for a city",
  parameters: WEATHER_TOOL.function.parameters,
};

// Gemini's answer to a weather question: two parallel calls, the first
// signed (Gemini signs only the first call of a batch).
const CALL_PARTS = [
  {
    functionCall: { name: "get_weather", args: { city: "Paris" } },
    thoughtSignature: SIGNATURE,
  },
  { functionCall: { name: "get_weather", args: { city: "Rome" } } },
];

const USAGE = {
  promptTokenCount: 20,
  candidatesTokenCount: 10,
  totalTokenCount: 30,
};

interface RecordedRequest {
  path: string;
  body: Record<string, any>;
}

interface VertexUpstream {
  baseUrl: string;
  received: RecordedRequest[];
  close(): Promise<void>;
}

function hasFunctionResponse(body: Record<string, any>): boolean {
  return (body.contents ?? []).some((c: any) =>
    (c.parts ?? []).some((p: any) => p.functionResponse !== undefined),
  );
}

/**
 * A Vertex Gemini upstream. A request whose history already carries
 * function responses gets the final text answer; any other gets the two
 * parallel weather calls. The streaming route sends each call in its
 * own frame and the finish reason on a third, the way Gemini streams a
 * tool-calling turn.
 */
async function startVertexUpstream(): Promise<VertexUpstream> {
  const received: RecordedRequest[] = [];
  const server: Server = createServer((req, res) => {
    res.on("error", () => {});
    const chunks: Buffer[] = [];
    req.on("data", (c: Buffer) => chunks.push(c));
    req.on("end", () => {
      const path = (req.url ?? "/").split("?")[0];
      const body = JSON.parse(Buffer.concat(chunks).toString("utf8"));
      received.push({ path, body });
      const answered = hasFunctionResponse(body);
      res.statusCode = 200;
      if (path.endsWith(":streamGenerateContent")) {
        res.setHeader("content-type", "text/event-stream");
        const frames = answered
          ? [
              {
                candidates: [
                  {
                    content: { role: "model", parts: [{ text: FINAL_ANSWER }] },
                    finishReason: "STOP",
                  },
                ],
                usageMetadata: USAGE,
              },
            ]
          : [
              { candidates: [{ content: { role: "model", parts: [CALL_PARTS[0]] } }] },
              { candidates: [{ content: { role: "model", parts: [CALL_PARTS[1]] } }] },
              {
                candidates: [
                  { content: { role: "model", parts: [{ text: "" }] }, finishReason: "STOP" },
                ],
                usageMetadata: USAGE,
              },
            ];
        res.end(frames.map((f) => `data: ${JSON.stringify(f)}\r\n\r\n`).join(""));
        return;
      }
      res.setHeader("content-type", "application/json");
      res.end(
        JSON.stringify({
          candidates: [
            {
              content: {
                role: "model",
                parts: answered ? [{ text: FINAL_ANSWER }] : CALL_PARTS,
              },
              finishReason: "STOP",
            },
          ],
          usageMetadata: USAGE,
        }),
      );
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const addr = server.address();
  if (addr === null || typeof addr === "string") throw new Error("no port");
  return {
    baseUrl: `http://127.0.0.1:${addr.port}`,
    received,
    close: () =>
      new Promise<void>((resolve, reject) =>
        server.close((e) => (e ? reject(e) : resolve())),
      ),
  };
}

function post(app: SpawnedApp, path: string, body: unknown): Promise<Response> {
  return fetch(`${app.proxyUrl}${path}`, {
    method: "POST",
    headers: {
      authorization: `Bearer ${CALLER_PLAINTEXT}`,
      "x-api-key": CALLER_PLAINTEXT,
      "content-type": "application/json",
    },
    body: JSON.stringify(body),
  });
}

/** The JSON payloads of every `data:` line of an SSE body. */
function sseData(raw: string): any[] {
  return raw
    .split(/\r?\n/)
    .filter((l) => l.startsWith("data: ") && !l.includes("[DONE]"))
    .map((l) => JSON.parse(l.slice(6)));
}

describe("function calling on Vertex's native Gemini wire", () => {
  let app: SpawnedApp | undefined;
  let upstream: VertexUpstream | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startVertexUpstream();
    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);
    const pk = await seed.createProviderKey({
      display_name: "vertex-fc-pk",
      provider: "google",
      adapter: "vertex",
      secret: JSON.stringify({
        access_token: "ya29.function-calling-e2e",
        project: "proj-e2e",
        region: "us-central1",
      }),
      api_base: upstream.baseUrl,
    });
    await seed.createModel({
      display_name: "vertex-fc-gemini",
      provider: "google",
      model_name: "gemini-2.5-flash",
      provider_key_id: pk.id,
    });
    await seed.createModel({
      display_name: "vertex-fc-gemini-3",
      provider: "google",
      model_name: "gemini-3-pro-preview",
      provider_key_id: pk.id,
    });
    // Seeded last, so this key authenticating implies the whole seed set
    // has reached the gateway's snapshot.
    await seed.createApiKey({ key_hash: CALLER_KEY_HASH, allowed_models: ["*"] });
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${CALLER_PLAINTEXT}` },
      });
      await res.text();
      return res.status === 200;
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  function lastUpstreamBody(): Record<string, any> {
    const last = upstream!.received.at(-1);
    expect(last, "upstream received no request").toBeDefined();
    return last!.body;
  }

  test("chat: tools and tool_choice reach Gemini, its calls come back as tool_calls", async (ctx) => {
    if (!etcdReachable || !app || !upstream) return ctx.skip();
    const res = await post(app, "/v1/chat/completions", {
      model: "vertex-fc-gemini",
      messages: [{ role: "user", content: "Weather in Paris and Rome?" }],
      tools: [WEATHER_TOOL],
      tool_choice: "required",
    });
    expect(res.status).toBe(200);
    const body = (await res.json()) as any;

    const sent = lastUpstreamBody();
    expect(upstream.received.at(-1)!.path).toContain("gemini-2.5-flash:generateContent");
    expect(sent.tools).toHaveLength(1);
    const declarations = sent.tools[0].functionDeclarations;
    expect(declarations).toHaveLength(1);
    expect(declarations[0].name).toBe("get_weather");
    expect(declarations[0].parameters.properties.city.type).toMatch(/^string$/i);
    expect(sent.toolConfig.functionCallingConfig.mode).toBe("ANY");

    const choice = body.choices[0];
    expect(choice.finish_reason).toBe("tool_calls");
    const calls = choice.message.tool_calls;
    expect(calls).toHaveLength(2);
    expect(calls.map((c: any) => c.function.name)).toEqual(["get_weather", "get_weather"]);
    expect(calls.map((c: any) => JSON.parse(c.function.arguments))).toEqual([
      { city: "Paris" },
      { city: "Rome" },
    ]);
    expect(calls.every((c: any) => c.type === "function" && c.id)).toBe(true);
    expect(calls[0].id).not.toBe(calls[1].id);
  });

  test("chat: a named tool_choice restricts Gemini to that function", async (ctx) => {
    if (!etcdReachable || !app || !upstream) return ctx.skip();
    const res = await post(app, "/v1/chat/completions", {
      model: "vertex-fc-gemini",
      messages: [{ role: "user", content: "Weather in Paris?" }],
      tools: [WEATHER_TOOL],
      tool_choice: { type: "function", function: { name: "get_weather" } },
    });
    expect(res.status).toBe(200);
    await res.text();
    expect(lastUpstreamBody().toolConfig.functionCallingConfig).toEqual({
      mode: "ANY",
      allowedFunctionNames: ["get_weather"],
    });
  });

  test("chat streaming: each call is its own indexed delta, the turn ends as tool_calls", async (ctx) => {
    if (!etcdReachable || !app || !upstream) return ctx.skip();
    const res = await post(app, "/v1/chat/completions", {
      model: "vertex-fc-gemini",
      messages: [{ role: "user", content: "Weather in Paris and Rome?" }],
      tools: [WEATHER_TOOL],
      stream: true,
    });
    expect(res.status).toBe(200);
    const frames = sseData(await res.text());
    expect(lastUpstreamBody().tools[0].functionDeclarations[0].name).toBe("get_weather");

    const deltas = frames.flatMap((f) => f.choices?.[0]?.delta?.tool_calls ?? []);
    const byIndex = new Map<number, { name: string; args: string; ids: string[] }>();
    for (const d of deltas) {
      expect(typeof d.index).toBe("number");
      const entry = byIndex.get(d.index) ?? { name: "", args: "", ids: [] };
      entry.name += d.function?.name ?? "";
      entry.args += d.function?.arguments ?? "";
      if (d.id) entry.ids.push(d.id);
      byIndex.set(d.index, entry);
    }
    expect(byIndex.size).toBe(2);
    const calls = [...byIndex.values()];
    expect(calls.map((c) => c.name)).toEqual(["get_weather", "get_weather"]);
    expect(calls.map((c) => JSON.parse(c.args))).toEqual([{ city: "Paris" }, { city: "Rome" }]);
    expect(calls.every((c) => c.ids.length === 1)).toBe(true);
    const finishes = frames.map((f) => f.choices?.[0]?.finish_reason).filter(Boolean);
    expect(finishes).toEqual(["tool_calls"]);
  });

  test("messages: Gemini's calls come back as tool_use blocks", async (ctx) => {
    if (!etcdReachable || !app || !upstream) return ctx.skip();
    const res = await post(app, "/v1/messages", {
      model: "vertex-fc-gemini",
      max_tokens: 256,
      messages: [{ role: "user", content: "Weather in Paris and Rome?" }],
      tools: [ANTHROPIC_WEATHER_TOOL],
      tool_choice: { type: "any" },
    });
    expect(res.status).toBe(200);
    const body = (await res.json()) as any;
    const sent = lastUpstreamBody();
    expect(sent.tools[0].functionDeclarations[0].name).toBe("get_weather");
    expect(sent.toolConfig.functionCallingConfig.mode).toBe("ANY");

    expect(body.stop_reason).toBe("tool_use");
    const uses = body.content.filter((b: any) => b.type === "tool_use");
    expect(uses.map((u: any) => [u.name, u.input])).toEqual([
      ["get_weather", { city: "Paris" }],
      ["get_weather", { city: "Rome" }],
    ]);
    expect(uses[0].id).not.toBe(uses[1].id);
  });

  test("messages streaming: each call opens its own tool_use block", async (ctx) => {
    if (!etcdReachable || !app || !upstream) return ctx.skip();
    const res = await post(app, "/v1/messages", {
      model: "vertex-fc-gemini",
      max_tokens: 256,
      stream: true,
      messages: [{ role: "user", content: "Weather in Paris and Rome?" }],
      tools: [ANTHROPIC_WEATHER_TOOL],
    });
    expect(res.status).toBe(200);
    const events = sseData(await res.text());
    const starts = events.filter(
      (e) => e.type === "content_block_start" && e.content_block?.type === "tool_use",
    );
    expect(starts.map((s) => s.content_block.name)).toEqual(["get_weather", "get_weather"]);
    expect(new Set(starts.map((s) => s.index)).size).toBe(2);
    const inputs = new Map<number, string>();
    for (const e of events) {
      if (e.type === "content_block_delta" && e.delta?.type === "input_json_delta") {
        inputs.set(e.index, (inputs.get(e.index) ?? "") + e.delta.partial_json);
      }
    }
    expect(starts.map((s) => JSON.parse(inputs.get(s.index) ?? ""))).toEqual([
      { city: "Paris" },
      { city: "Rome" },
    ]);
    const stop = events.find((e) => e.type === "message_delta");
    expect(stop?.delta?.stop_reason).toBe("tool_use");
  });

  test("responses: Gemini's calls come back as function_call items", async (ctx) => {
    if (!etcdReachable || !app || !upstream) return ctx.skip();
    const res = await post(app, "/v1/responses", {
      model: "vertex-fc-gemini",
      input: "Weather in Paris and Rome?",
      tools: [RESPONSES_WEATHER_TOOL],
      tool_choice: "required",
    });
    expect(res.status).toBe(200);
    const body = (await res.json()) as any;
    const sent = lastUpstreamBody();
    expect(sent.tools[0].functionDeclarations[0].name).toBe("get_weather");
    expect(sent.toolConfig.functionCallingConfig.mode).toBe("ANY");

    const calls = body.output.filter((o: any) => o.type === "function_call");
    expect(calls.map((c: any) => [c.name, JSON.parse(c.arguments)])).toEqual([
      ["get_weather", { city: "Paris" }],
      ["get_weather", { city: "Rome" }],
    ]);
    expect(calls[0].call_id).not.toBe(calls[1].call_id);
  });

  test("responses streaming: the completed response carries both function_call items", async (ctx) => {
    if (!etcdReachable || !app || !upstream) return ctx.skip();
    const res = await post(app, "/v1/responses", {
      model: "vertex-fc-gemini",
      input: "Weather in Paris and Rome?",
      tools: [RESPONSES_WEATHER_TOOL],
      stream: true,
    });
    expect(res.status).toBe(200);
    const events = sseData(await res.text());
    const completed = events.find((e) => e.type === "response.completed");
    expect(completed).toBeDefined();
    const calls = completed.response.output.filter((o: any) => o.type === "function_call");
    expect(calls.map((c: any) => [c.name, JSON.parse(c.arguments)])).toEqual([
      ["get_weather", { city: "Paris" }],
      ["get_weather", { city: "Rome" }],
    ]);
  });

  test("chat: a tool round trip goes back to Gemini as functionCall and functionResponse parts", async (ctx) => {
    if (!etcdReachable || !app || !upstream) return ctx.skip();
    const question = { role: "user", content: "Weather in Paris and Rome?" };
    const first = await post(app, "/v1/chat/completions", {
      model: "vertex-fc-gemini",
      messages: [question],
      tools: [WEATHER_TOOL],
    });
    expect(first.status).toBe(200);
    const assistant = ((await first.json()) as any).choices[0].message;
    expect(assistant.tool_calls).toHaveLength(2);

    const second = await post(app, "/v1/chat/completions", {
      model: "vertex-fc-gemini",
      messages: [
        question,
        assistant,
        { role: "tool", tool_call_id: assistant.tool_calls[0].id, content: "sunny" },
        { role: "tool", tool_call_id: assistant.tool_calls[1].id, content: '{"temp":21}' },
      ],
      tools: [WEATHER_TOOL],
    });
    expect(second.status).toBe(200);
    const answer = (await second.json()) as any;
    expect(answer.choices[0].message.content).toBe(FINAL_ANSWER);
    expect(answer.choices[0].finish_reason).toBe("stop");

    const sent = lastUpstreamBody();
    expect(sent.contents.map((c: any) => c.role)).toEqual(["user", "model", "user"]);
    const model = sent.contents[1].parts;
    expect(model.map((p: any) => [p.functionCall.name, p.functionCall.args])).toEqual([
      ["get_weather", { city: "Paris" }],
      ["get_weather", { city: "Rome" }],
    ]);
    // Gemini rejects a replayed call without the signature it was sent with.
    expect(model[0].thoughtSignature).toBe(SIGNATURE);
    expect(
      sent.contents[2].parts.map((p: any) => [p.functionResponse.name, p.functionResponse.response]),
    ).toEqual([
      ["get_weather", { content: "sunny" }],
      ["get_weather", { temp: 21 }],
    ]);
  });

  test.for(["vertex-fc-gemini", "vertex-fc-gemini-3"])(
    "%s: replayed calls and responses carry the call id",
    async (modelName, ctx) => {
    if (!etcdReachable || !app || !upstream) return ctx.skip();
    const question = { role: "user", content: "Weather in Paris and Rome?" };
    const first = await post(app, "/v1/chat/completions", {
      model: modelName,
      messages: [question],
      tools: [WEATHER_TOOL],
    });
    expect(first.status).toBe(200);
    const assistant = ((await first.json()) as any).choices[0].message;
    const second = await post(app, "/v1/chat/completions", {
      model: modelName,
      messages: [
        question,
        assistant,
        { role: "tool", tool_call_id: assistant.tool_calls[0].id, content: "sunny" },
        { role: "tool", tool_call_id: assistant.tool_calls[1].id, content: "windy" },
      ],
      tools: [WEATHER_TOOL],
    });
    expect(second.status).toBe(200);
    await second.text();
    const sent = lastUpstreamBody();
    const callIds = sent.contents[1].parts.map((p: any) => p.functionCall.id);
    const responseIds = sent.contents[2].parts.map((p: any) => p.functionResponse.id);
    expect(callIds.every((id: unknown) => typeof id === "string" && id.length > 0)).toBe(true);
    expect(responseIds).toEqual(callIds);
    expect(sent.contents[1].parts[0].thoughtSignature).toBe(SIGNATURE);
  },
  );

  test("messages: a tool_result history goes back as functionResponse parts", async (ctx) => {
    if (!etcdReachable || !app || !upstream) return ctx.skip();
    const question = { role: "user", content: "Weather in Paris and Rome?" };
    const first = await post(app, "/v1/messages", {
      model: "vertex-fc-gemini",
      max_tokens: 256,
      messages: [question],
      tools: [ANTHROPIC_WEATHER_TOOL],
    });
    expect(first.status).toBe(200);
    const turn = (await first.json()) as any;
    const uses = turn.content.filter((b: any) => b.type === "tool_use");
    const second = await post(app, "/v1/messages", {
      model: "vertex-fc-gemini",
      max_tokens: 256,
      tools: [ANTHROPIC_WEATHER_TOOL],
      messages: [
        question,
        { role: "assistant", content: turn.content },
        {
          role: "user",
          content: [
            { type: "tool_result", tool_use_id: uses[0].id, content: "sunny" },
            { type: "tool_result", tool_use_id: uses[1].id, content: "windy" },
          ],
        },
      ],
    });
    expect(second.status).toBe(200);
    const answer = (await second.json()) as any;
    expect(answer.content.find((b: any) => b.type === "text")?.text).toBe(FINAL_ANSWER);

    const sent = lastUpstreamBody();
    const model = sent.contents.find((c: any) => c.role === "model");
    expect(model.parts.map((p: any) => p.functionCall?.args)).toEqual([
      { city: "Paris" },
      { city: "Rome" },
    ]);
    expect(model.parts[0].thoughtSignature).toBe(SIGNATURE);
    const responses = sent.contents.flatMap((c: any) =>
      (c.parts ?? []).filter((p: any) => p.functionResponse),
    );
    expect(
      responses.map((p: any) => [p.functionResponse.name, p.functionResponse.response]),
    ).toEqual([
      ["get_weather", { content: "sunny" }],
      ["get_weather", { content: "windy" }],
    ]);
  });

  test("chat: ids handed to the client are portable", async (ctx) => {
    if (!etcdReachable || !app || !upstream) return ctx.skip();
    const res = await post(app, "/v1/chat/completions", {
      model: "vertex-fc-gemini",
      messages: [{ role: "user", content: "Weather in Paris and Rome?" }],
      tools: [WEATHER_TOOL],
    });
    expect(res.status).toBe(200);
    const calls = ((await res.json()) as any).choices[0].message.tool_calls;
    for (const call of calls) expect(call.id).toMatch(/^[A-Za-z0-9_-]+$/);
  });

  test("chat: a schema the typed form cannot express goes as parametersJsonSchema", async (ctx) => {
    if (!etcdReachable || !app || !upstream) return ctx.skip();
    const parameters = {
      type: "object",
      properties: {
        place: { oneOf: [{ type: "string" }, { type: "integer" }] },
      },
      required: ["place"],
    };
    const res = await post(app, "/v1/chat/completions", {
      model: "vertex-fc-gemini",
      messages: [{ role: "user", content: "Weather in Paris?" }],
      tools: [{ type: "function", function: { name: "get_weather", parameters } }],
    });
    expect(res.status).toBe(200);
    await res.text();
    const declaration = lastUpstreamBody().tools[0].functionDeclarations[0];
    expect(declaration.parametersJsonSchema).toEqual(parameters);
    expect(declaration.parameters).toBeUndefined();
  });

  test.for<[string, Record<string, unknown>]>([
    ["legacy functions", { functions: [WEATHER_TOOL.function] }],
    ["legacy function_call", { tools: [WEATHER_TOOL], function_call: "auto" }],
    ["a hosted tool", { tools: [{ type: "web_search_preview" }] }],
    [
      "an unnamed tool_choice object",
      { tools: [WEATHER_TOOL], tool_choice: { type: "allowed_tools", allowed_tools: {} } },
    ],
  ])("chat: %s is refused, not dropped", async ([, extra], ctx) => {
    if (!etcdReachable || !app || !upstream) return ctx.skip();
    const before = upstream.received.length;
    const res = await post(app, "/v1/chat/completions", {
      model: "vertex-fc-gemini",
      messages: [{ role: "user", content: "Weather in Paris?" }],
      ...extra,
    });
    expect(res.status).toBe(400);
    const body = (await res.json()) as any;
    expect(body.error.type).toBe("invalid_request_error");
    expect(upstream.received.length).toBe(before);
  });
});

// A tool call id minted for a Gemini answer carries that answer's thought
// signature. When the conversation continues on another provider the id
// has to be one that provider accepts: OpenAI caps ids at 64 characters,
// Bedrock also restricts them to [A-Za-z0-9_-].
describe("a Gemini tool call id continuing on another provider", () => {
  let app: SpawnedApp | undefined;
  let vertex: VertexUpstream | undefined;
  let openai: OpenAiUpstream | undefined;
  let other: { baseUrl: string; received: RecordedRequest[]; close(): Promise<void> } | undefined;
  let etcdReachable = false;
  const moderated: string[] = [];

  // Answers the moderation endpoint, the Bedrock Converse, Anthropic Messages (and count_tokens)
  // and OpenAI Responses routes, recording what each received.
  async function startOtherUpstream() {
    const received: RecordedRequest[] = [];
    const server: Server = createServer((req, res) => {
      const chunks: Buffer[] = [];
      req.on("data", (c: Buffer) => chunks.push(c));
      req.on("end", () => {
        const path = (req.url ?? "/").split("?")[0];
        received.push({ path, body: JSON.parse(Buffer.concat(chunks).toString("utf8")) });
        res.setHeader("content-type", "application/json");
        let reply: unknown;
        if (path.endsWith("/moderations")) {
          const input = String((received.at(-1)!.body as any).input ?? "");
          reply = {
            id: "modr-1",
            model: "omni-moderation-latest",
            results: [{ flagged: false, categories: {}, category_scores: {} }],
          };
          moderated.push(input);
        } else if (path.endsWith("/converse")) {
          reply = {
            output: { message: { role: "assistant", content: [{ text: FINAL_ANSWER }] } },
            stopReason: "end_turn",
            usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 },
            metrics: { latencyMs: 1 },
          };
        } else if (path.endsWith("/count_tokens")) {
          reply = { input_tokens: 12 };
        } else if (path.endsWith("/messages")) {
          reply = {
            id: "msg_1",
            type: "message",
            role: "assistant",
            model: "claude-sonnet-4-5",
            content: [{ type: "text", text: FINAL_ANSWER }],
            stop_reason: "end_turn",
            usage: { input_tokens: 1, output_tokens: 1 },
          };
        } else {
          reply = {
            id: "resp_1",
            object: "response",
            status: "completed",
            model: "gpt-4o",
            output: [
              {
                type: "message",
                id: "msg_1",
                role: "assistant",
                status: "completed",
                content: [{ type: "output_text", text: FINAL_ANSWER, annotations: [] }],
              },
            ],
            usage: { input_tokens: 1, output_tokens: 1, total_tokens: 2 },
          };
        }
        res.end(JSON.stringify(reply));
      });
    });
    await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
    const addr = server.address() as { port: number };
    return {
      baseUrl: `http://127.0.0.1:${addr.port}`,
      received,
      close: () => new Promise<void>((resolve) => server.close(() => resolve())),
    };
  }

  function lastAt(suffix: string): Record<string, any> {
    const hit = [...other!.received].reverse().find((r) => r.path.endsWith(suffix));
    expect(hit, `nothing reached ${suffix}`).toBeDefined();
    return hit!.body;
  }

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    vertex = await startVertexUpstream();
    openai = await startOpenAiUpstream({
      nonStreamBody: {
        id: "chatcmpl-1",
        object: "chat.completion",
        created: 1,
        model: "gpt-4o",
        choices: [{ index: 0, message: { role: "assistant", content: FINAL_ANSWER }, finish_reason: "stop" }],
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
      },
    });
    other = await startOtherUpstream();

    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);
    const vertexPk = await seed.createProviderKey({
      display_name: "vertex-portable-pk",
      provider: "google",
      adapter: "vertex",
      secret: JSON.stringify({ access_token: "ya29.portable", project: "p", region: "us-central1" }),
      api_base: vertex.baseUrl,
    });
    await seed.createModel({
      display_name: "portable-gemini",
      provider: "google",
      model_name: "gemini-2.5-flash",
      provider_key_id: vertexPk.id,
    });
    // A moderation guardrail scans the whole tool-call payload; what it is
    // sent is recorded.
    const guarded = await seed.createModel({
      display_name: "portable-gemini-guarded",
      provider: "google",
      model_name: "gemini-2.5-flash",
      provider_key_id: vertexPk.id,
    });
    const guardrail = await seed.createGuardrail(
      {
        name: "tool-call-moderation",
        enabled: true,
        kind: "openai_moderation",
        hook_point: "both",
        fail_open: false,
        api_key: "sk-moderation-key",
        endpoint: other.baseUrl,
      },
      { attach: false },
    );
    await seed.attachGuardrailToModel(guardrail.id as string, guarded.id as string);
    const openaiPk = await seed.createProviderKey({
      display_name: "portable-openai-pk",
      secret: "sk-openai-mock",
      api_base: `${openai.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "portable-gpt",
      provider: "openai",
      model_name: "gpt-4o",
      provider_key_id: openaiPk.id,
    });
    const responsesPk = await seed.createProviderKey({
      display_name: "portable-responses-pk",
      secret: "sk-openai-mock",
      api_base: `${other.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "portable-gpt-responses",
      provider: "openai",
      model_name: "gpt-4o",
      provider_key_id: responsesPk.id,
    });
    const anthropicPk = await seed.createProviderKey({
      display_name: "portable-anthropic-pk",
      provider: "anthropic",
      adapter: "anthropic",
      secret: "sk-ant-mock",
      api_base: other.baseUrl,
    });
    await seed.createModel({
      display_name: "portable-claude",
      provider: "anthropic",
      model_name: "claude-sonnet-4-5",
      provider_key_id: anthropicPk.id,
    });
    const bedrockPk = await seed.createProviderKey({
      display_name: "portable-bedrock-pk",
      provider: "bedrock",
      adapter: "bedrock",
      secret: JSON.stringify({
        access_key_id: "AKIA-portable",
        secret_access_key: "sk-portable",
        region: "us-west-2",
      }),
      api_base: other.baseUrl,
    });
    await seed.createModel({
      display_name: "portable-nova",
      provider: "bedrock",
      model_name: "amazon.nova-pro-v1:0",
      provider_key_id: bedrockPk.id,
    });
    await seed.createApiKey({ key_hash: CALLER_KEY_HASH, allowed_models: ["*"] });
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${CALLER_PLAINTEXT}` },
      });
      await res.text();
      return res.status === 200;
    });
  });

  afterAll(async () => {
    await app?.exit();
    await vertex?.close();
    await openai?.close();
    await other?.close();
  });

  const QUESTION = { role: "user", content: "Weather in Paris and Rome?" };

  async function chatHistory(model = "portable-gemini"): Promise<any[]> {
    const first = await post(app!, "/v1/chat/completions", {
      model,
      messages: [QUESTION],
      tools: [WEATHER_TOOL],
    });
    expect(first.status).toBe(200);
    const assistant = ((await first.json()) as any).choices[0].message;
    expect(assistant.tool_calls).toHaveLength(2);
    return [
      QUESTION,
      assistant,
      ...assistant.tool_calls.map((c: any) => ({ role: "tool", tool_call_id: c.id, content: "sunny" })),
    ];
  }

  async function messagesHistory(): Promise<any[]> {
    const first = await post(app!, "/v1/messages", {
      model: "portable-gemini",
      max_tokens: 256,
      messages: [QUESTION],
      tools: [ANTHROPIC_WEATHER_TOOL],
    });
    expect(first.status).toBe(200);
    const turn = (await first.json()) as any;
    const uses = turn.content.filter((b: any) => b.type === "tool_use");
    expect(uses).toHaveLength(2);
    return [
      QUESTION,
      { role: "assistant", content: turn.content },
      {
        role: "user",
        content: uses.map((u: any) => ({ type: "tool_result", tool_use_id: u.id, content: "sunny" })),
      },
    ];
  }

  function expectPortable(ids: string[]) {
    expect(ids).toHaveLength(4);
    for (const id of ids) {
      expect(id.length).toBeLessThanOrEqual(64);
      expect(id).toMatch(/^[A-Za-z0-9_-]+$/);
    }
    // Each call and the result answering it still agree.
    expect(ids.slice(0, 2)).toEqual(ids.slice(2));
  }

  function anthropicIds(body: Record<string, any>): string[] {
    const blocks = body.messages.flatMap((m: any) => (Array.isArray(m.content) ? m.content : []));
    return [
      ...blocks.filter((b: any) => b.type === "tool_use").map((b: any) => b.id),
      ...blocks.filter((b: any) => b.type === "tool_result").map((b: any) => b.tool_use_id),
    ];
  }

  test("an OpenAI upstream receives ids it accepts", async (ctx) => {
    if (!etcdReachable || !app || !openai) return ctx.skip();
    const res = await post(app, "/v1/chat/completions", {
      model: "portable-gpt",
      messages: await chatHistory(),
      tools: [WEATHER_TOOL],
    });
    expect(res.status).toBe(200);
    await res.text();
    const sent = JSON.parse(openai.receivedRequests.at(-1)!.body);
    expectPortable([
      ...sent.messages[1].tool_calls.map((c: any) => c.id),
      ...sent.messages.slice(2).map((m: any) => m.tool_call_id),
    ]);
  });

  test("a Bedrock upstream receives ids it accepts", async (ctx) => {
    if (!etcdReachable || !app || !other) return ctx.skip();
    const res = await post(app, "/v1/chat/completions", {
      model: "portable-nova",
      messages: await chatHistory(),
      tools: [WEATHER_TOOL],
    });
    expect(res.status).toBe(200);
    await res.text();
    const blocks = lastAt("/converse").messages.flatMap((m: any) => m.content);
    expectPortable([
      ...blocks.filter((b: any) => b.toolUse).map((b: any) => b.toolUse.toolUseId),
      ...blocks.filter((b: any) => b.toolResult).map((b: any) => b.toolResult.toolUseId),
    ]);
  });

  test("an Anthropic upstream reached by chat receives ids it accepts", async (ctx) => {
    if (!etcdReachable || !app || !other) return ctx.skip();
    const res = await post(app, "/v1/chat/completions", {
      model: "portable-claude",
      max_tokens: 256,
      messages: await chatHistory(),
      tools: [WEATHER_TOOL],
    });
    expect(res.status).toBe(200);
    await res.text();
    expectPortable(anthropicIds(lastAt("/messages")));
  });

  test("the native /v1/messages and count_tokens paths forward ids Anthropic accepts", async (ctx) => {
    if (!etcdReachable || !app || !other) return ctx.skip();
    const messages = await messagesHistory();
    const res = await post(app, "/v1/messages", {
      model: "portable-claude",
      max_tokens: 256,
      messages,
      tools: [ANTHROPIC_WEATHER_TOOL],
    });
    expect(res.status).toBe(200);
    await res.text();
    expectPortable(anthropicIds(lastAt("/messages")));

    const count = await post(app, "/v1/messages/count_tokens", {
      model: "portable-claude",
      messages,
      tools: [ANTHROPIC_WEATHER_TOOL],
    });
    expect(count.status).toBe(200);
    await count.text();
    expectPortable(anthropicIds(lastAt("/count_tokens")));
  });

  test("the native /v1/responses path forwards call ids OpenAI accepts", async (ctx) => {
    if (!etcdReachable || !app || !other) return ctx.skip();
    const first = await post(app, "/v1/responses", {
      model: "portable-gemini",
      input: "Weather in Paris and Rome?",
      tools: [RESPONSES_WEATHER_TOOL],
    });
    expect(first.status).toBe(200);
    const calls = ((await first.json()) as any).output.filter((o: any) => o.type === "function_call");
    expect(calls).toHaveLength(2);
    const res = await post(app, "/v1/responses", {
      model: "portable-gpt-responses",
      input: [
        { role: "user", content: "Weather in Paris and Rome?" },
        ...calls.map((c: any) => ({
          type: "function_call",
          call_id: c.call_id,
          name: c.name,
          arguments: c.arguments,
        })),
        ...calls.map((c: any) => ({ type: "function_call_output", call_id: c.call_id, output: "sunny" })),
      ],
      tools: [RESPONSES_WEATHER_TOOL],
    });
    expect(res.status).toBe(200);
    await res.text();
    expectPortable(lastAt("/responses").input.filter((i: any) => i.call_id).map((i: any) => i.call_id));
  });

  test("guardrails scan tool calls without their ids", async (ctx) => {
    if (!etcdReachable || !app || !vertex) return ctx.skip();
    // Turn 1 runs the output scan over the signed calls, turn 2 the input
    // scan over the history carrying them.
    moderated.length = 0;
    const messages = await chatHistory("portable-gemini-guarded");
    const res = await post(app, "/v1/chat/completions", {
      model: "portable-gemini-guarded",
      messages,
      tools: [WEATHER_TOOL],
    });
    expect(res.status).toBe(200);
    await res.text();
    const ids = messages[1].tool_calls.map((c: any) => c.id);
    const withCalls = moderated.filter((text) => text.includes("Rome"));
    // The calls were scanned on the way out and on the way back in ...
    expect(withCalls.length).toBeGreaterThanOrEqual(2);
    // ... without their ids.
    for (const text of withCalls) for (const id of ids) expect(text).not.toContain(id);
  });
});
