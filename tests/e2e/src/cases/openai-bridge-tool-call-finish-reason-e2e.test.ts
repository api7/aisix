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

// E2E: a turn that called tools finishes as a tool-calling turn even when an
// OpenAI-compatible upstream ends it with the generic `stop`.
//
// Gemini's OpenAI-compatible endpoint does exactly that: the stream below
// follows one captured from Google AI Studio (`gemini-3.6-flash`, two
// parallel calls), where each chunk carries one whole call with no `index`
// and the turn closes with `finish_reason: "stop"` (the thought signature is
// replaced with a marker). A client that runs tools on `tool_calls`, or on
// the `tool_use` stop reason of `/v1/messages`, would otherwise stop instead
// of calling the tools. Other terminal reasons are not rewritten: a turn cut
// off by the token limit still says so.

const CALLER_PLAINTEXT = "sk-tool-call-finish-reason";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");
const TRUNCATE = "truncated tool call please";

function chunk(delta: Record<string, unknown>, finishReason?: string): string {
  return JSON.stringify({
    choices: [
      {
        delta: { role: "assistant", ...delta },
        ...(finishReason ? { finish_reason: finishReason } : {}),
        index: 0,
      },
    ],
    created: 1791854951,
    id: "ZYnNas-XL_fyqtsPzceKiA8",
    model: "gemini-3.6-flash",
    object: "chat.completion.chunk",
  });
}

function call(id: string, location: string, extra: Record<string, unknown> = {}) {
  return {
    ...extra,
    function: { arguments: JSON.stringify({ location }), name: "get_weather" },
    id,
    type: "function",
  };
}

const PARIS = call("call_121133", "Paris", {
  extra_content: { google: { thought_signature: "THOUGHT_SIGNATURE_FIXTURE" } },
});
const ROME = call("call_121134", "Rome");

function sse(events: string[]): string {
  return [...events, "[DONE]"].map((e) => `data: ${e}\n\n`).join("");
}

const PARALLEL_STREAM = sse([
  chunk({ tool_calls: [PARIS] }),
  chunk({ tool_calls: [ROME] }),
  chunk({}, "stop"),
]);
const TRUNCATED_STREAM = sse([chunk({ tool_calls: [PARIS] }), chunk({}, "length")]);

const PARALLEL_REPLY = {
  id: "ZYnNas-XL_fyqtsPzceKiA9",
  object: "chat.completion",
  created: 1791854951,
  model: "gemini-3.6-flash",
  choices: [
    {
      index: 0,
      message: { role: "assistant", content: null, tool_calls: [PARIS, ROME] },
      finish_reason: "stop",
    },
  ],
  usage: { prompt_tokens: 30, completion_tokens: 20, total_tokens: 50 },
};

const PROMPT = "Weather in Paris and Rome? Call get_weather for each, in parallel.";
const WEATHER_PARAMS = {
  type: "object",
  properties: { location: { type: "string" } },
  required: ["location"],
};
const CHAT_TOOLS = [{ type: "function", function: { name: "get_weather", parameters: WEATHER_PARAMS } }];
const MESSAGES_TOOLS = [{ name: "get_weather", input_schema: WEATHER_PARAMS }];
const RESPONSES_TOOLS = [{ type: "function", name: "get_weather", parameters: WEATHER_PARAMS }];

// One Model per provider that reaches the upstream through the OpenAI bridge.
const MODELS = [
  ["google", "finish-gemini", "gemini-3.6-flash"],
  ["deepseek", "finish-deepseek", "deepseek-chat"],
] as const;

describe("tool-calling turns ended with `stop` by an OpenAI-compatible upstream", () => {
  let app: SpawnedApp | undefined;
  let upstream: Server | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = createServer((req, res) => {
      let raw = "";
      req.on("data", (c) => (raw += c));
      req.on("end", () => {
        const body = JSON.parse(raw || "{}");
        if (body.stream) {
          res.writeHead(200, { "content-type": "text/event-stream" });
          res.end(raw.includes(TRUNCATE) ? TRUNCATED_STREAM : PARALLEL_STREAM);
        } else {
          res.writeHead(200, { "content-type": "application/json" });
          res.end(JSON.stringify(PARALLEL_REPLY));
        }
      });
    });
    await new Promise<void>((resolve) => upstream!.listen(0, "127.0.0.1", resolve));
    const addr = upstream.address();
    if (addr === null || typeof addr === "string") throw new Error("no port");

    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);
    for (const [provider, alias, name] of MODELS) {
      const pk = await seed.createProviderKey({
        display_name: `${alias}-pk`,
        provider,
        secret: "sk-finish-reason-mock",
        api_base: `http://127.0.0.1:${addr.port}`,
      });
      await seed.createModel({
        display_name: alias,
        provider,
        model_name: name,
        provider_key_id: pk.id,
      });
    }
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
    await new Promise<void>((resolve) => (upstream ? upstream.close(() => resolve()) : resolve()));
  });

  async function post(path: string, body: unknown): Promise<string> {
    const res = await fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "x-api-key": CALLER_PLAINTEXT,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
      },
      body: JSON.stringify(body),
    });
    const text = await res.text();
    expect(res.status, text).toBe(200);
    return text;
  }

  function sseData(text: string): any[] {
    return text
      .split("\n")
      .filter((l) => l.startsWith("data: ") && l !== "data: [DONE]")
      .map((l) => JSON.parse(l.slice("data: ".length)));
  }

  function chatRequest(model: string, prompt = PROMPT) {
    return { model, tools: CHAT_TOOLS, messages: [{ role: "user", content: prompt }] };
  }

  function messagesRequest(model: string, prompt = PROMPT) {
    return {
      model,
      max_tokens: 256,
      tools: MESSAGES_TOOLS,
      messages: [{ role: "user", content: prompt }],
    };
  }

  describe.for(MODELS)("%s", ([, model]) => {
    test("/v1/chat/completions reports tool_calls", async (ctx) => {
      if (!etcdReachable || !app) return ctx.skip();
      const plain = JSON.parse(await post("/v1/chat/completions", chatRequest(model)));
      expect(plain.choices[0].message.tool_calls).toHaveLength(2);
      expect(plain.choices[0].finish_reason).toBe("tool_calls");

      const chunks = sseData(await post("/v1/chat/completions", { ...chatRequest(model), stream: true }));
      const calls = chunks.flatMap((c) => c.choices?.[0]?.delta?.tool_calls ?? []);
      expect(calls.map((c: any) => c.id)).toEqual(["call_121133", "call_121134"]);
      const finishes = chunks.map((c) => c.choices?.[0]?.finish_reason).filter(Boolean);
      expect(finishes).toEqual(["tool_calls"]);
    });

    test("/v1/messages reports tool_use", async (ctx) => {
      if (!etcdReachable || !app) return ctx.skip();
      const plain = JSON.parse(await post("/v1/messages", messagesRequest(model)));
      expect(plain.content.filter((b: any) => b.type === "tool_use")).toHaveLength(2);
      expect(plain.stop_reason).toBe("tool_use");

      const events = sseData(await post("/v1/messages", { ...messagesRequest(model), stream: true }));
      const toolUses = events.filter(
        (e) => e.type === "content_block_start" && e.content_block?.type === "tool_use",
      );
      expect(toolUses).toHaveLength(2);
      const stops = events.filter((e) => e.type === "message_delta").map((e) => e.delta.stop_reason);
      expect(stops).toEqual(["tool_use"]);
    });

    test("/v1/responses still completes with both function calls", async (ctx) => {
      if (!etcdReachable || !app) return ctx.skip();
      const request = { model, tools: RESPONSES_TOOLS, input: PROMPT };
      const calls = (output: any[]) =>
        output.filter((i) => i.type === "function_call").map((i) => i.call_id);

      const plain = JSON.parse(await post("/v1/responses", request));
      expect(plain.status).toBe("completed");
      expect(calls(plain.output)).toEqual(["call_121133", "call_121134"]);

      const events = sseData(await post("/v1/responses", { ...request, stream: true }));
      const completed = events.find((e) => e.type === "response.completed");
      expect(completed?.response.status).toBe("completed");
      expect(calls(completed.response.output)).toEqual(["call_121133", "call_121134"]);
    });

    test("a tool-calling turn cut off by the token limit still reports the limit", async (ctx) => {
      if (!etcdReachable || !app) return ctx.skip();
      const chunks = sseData(
        await post("/v1/chat/completions", { ...chatRequest(model, TRUNCATE), stream: true }),
      );
      expect(chunks.map((c) => c.choices?.[0]?.finish_reason).filter(Boolean)).toEqual(["length"]);

      const events = sseData(
        await post("/v1/messages", { ...messagesRequest(model, TRUNCATE), stream: true }),
      );
      const stops = events.filter((e) => e.type === "message_delta").map((e) => e.delta.stop_reason);
      expect(stops).toEqual(["max_tokens"]);
    });
  });
});
