import { createHash } from "node:crypto";
import { createServer, type Server } from "node:http";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  spawnApp,
  waitConfigPropagation,
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

const SIGNATURE = "c2lnbmF0dXJlLWZvci1wYXJpcw==";
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
    expect(model.map((p: any) => p.functionCall)).toEqual([
      { name: "get_weather", args: { city: "Paris" } },
      { name: "get_weather", args: { city: "Rome" } },
    ]);
    // Gemini rejects a replayed call without the signature it was sent with.
    expect(model[0].thoughtSignature).toBe(SIGNATURE);
    expect(sent.contents[2].parts.map((p: any) => p.functionResponse)).toEqual([
      { name: "get_weather", response: { content: "sunny" } },
      { name: "get_weather", response: { temp: 21 } },
    ]);
  });

  test("Gemini 3: replayed calls and responses carry the call id", async (ctx) => {
    if (!etcdReachable || !app || !upstream) return ctx.skip();
    const question = { role: "user", content: "Weather in Paris and Rome?" };
    const first = await post(app, "/v1/chat/completions", {
      model: "vertex-fc-gemini-3",
      messages: [question],
      tools: [WEATHER_TOOL],
    });
    expect(first.status).toBe(200);
    const assistant = ((await first.json()) as any).choices[0].message;
    const second = await post(app, "/v1/chat/completions", {
      model: "vertex-fc-gemini-3",
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
  });

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
    expect(responses.map((p: any) => p.functionResponse)).toEqual([
      { name: "get_weather", response: { content: "sunny" } },
      { name: "get_weather", response: { content: "windy" } },
    ]);
  });
});
