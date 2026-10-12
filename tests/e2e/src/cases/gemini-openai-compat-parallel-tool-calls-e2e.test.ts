import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: parallel tool calls streamed by a Google AI Studio
// (`provider: "google"`) Model through Gemini's OpenAI-compatible
// `/chat/completions`.
//
// Gemini sends each call whole in its own chunk and gives the tool-call
// delta no `index` (chunks below are the shape captured from a live
// `gemini-3.6-flash` call; `extra_content` dropped). Every downstream
// surface groups streamed tool-call fragments by `index`, so both calls
// must come out as separate calls rather than one call whose arguments are
// `{"city":"Paris"}{"city":"Rome"}`.

const CALLER_PLAINTEXT = "sk-gemini-compat-parallel-tools-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");
const MODEL = "gemini-compat-parallel-tools";

function chunk(delta: Record<string, unknown>, finishReason?: string): string {
  return JSON.stringify({
    choices: [
      {
        delta: { role: "assistant", ...delta },
        index: 0,
        ...(finishReason ? { finish_reason: finishReason } : {}),
      },
    ],
    created: 1791799167,
    id: "fq_MarinFa2u_PUPtb3C4A0",
    model: "gemini-3.6-flash",
    object: "chat.completion.chunk",
  });
}

function toolCall(id: string, city: string) {
  return {
    tool_calls: [
      {
        function: { arguments: JSON.stringify({ city }), name: "get_weather" },
        id,
        type: "function",
      },
    ],
  };
}

const GEMINI_STREAM = [
  chunk(toolCall("call_59014", "Paris")),
  chunk(toolCall("call_59015", "Rome")),
  chunk({}, "stop"),
  "[DONE]",
];

const PROMPT = "What is the weather in Paris and in Rome?";
const WEATHER_PARAMS = {
  type: "object",
  properties: { city: { type: "string" } },
  required: ["city"],
};

describe("Gemini OpenAI-compatible streamed parallel tool calls", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream({ streamEvents: GEMINI_STREAM });
    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);
    const pk = await seed.createProviderKey({
      display_name: "gemini-compat-parallel-tools-pk",
      provider: "google",
      secret: "gemini-mock-key",
      api_base: upstream.baseUrl,
    });
    await seed.createModel({
      display_name: MODEL,
      provider: "google",
      model_name: "gemini-3.6-flash",
      provider_key_id: pk.id,
    });
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: [MODEL],
    });

    const proxy = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    await waitConfigPropagation(async () => {
      const res = await proxy.listModels();
      if (res.status !== 200) return false;
      const data = (res.body as { data?: Array<{ id?: string }> }).data ?? [];
      return data.some((m) => m.id === MODEL);
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  async function streamEvents(path: string, body: unknown): Promise<any[]> {
    const res = await fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
      },
      body: JSON.stringify(body),
    });
    const text = await res.text();
    expect(res.status, text).toBe(200);
    return text
      .split("\n")
      .filter((l) => l.startsWith("data: ") && l !== "data: [DONE]")
      .map((l) => JSON.parse(l.slice("data: ".length)));
  }

  test("/v1/messages emits one tool_use block per call", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }
    const events = await streamEvents("/v1/messages", {
      model: MODEL,
      max_tokens: 256,
      stream: true,
      messages: [{ role: "user", content: PROMPT }],
      tools: [{ name: "get_weather", input_schema: WEATHER_PARAMS }],
    });

    const blocks = new Map<number, { id: string; json: string }>();
    for (const e of events) {
      if (e.type === "content_block_start" && e.content_block.type === "tool_use") {
        blocks.set(e.index, { id: e.content_block.id, json: "" });
      } else if (e.type === "content_block_delta" && e.delta.type === "input_json_delta") {
        blocks.get(e.index)!.json += e.delta.partial_json;
      }
    }
    expect([...blocks.values()]).toEqual([
      { id: "call_59014", json: '{"city":"Paris"}' },
      { id: "call_59015", json: '{"city":"Rome"}' },
    ]);
  });

  test("/v1/responses emits one function_call item per call", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }
    const events = await streamEvents("/v1/responses", {
      model: MODEL,
      stream: true,
      input: PROMPT,
      tools: [{ type: "function", name: "get_weather", parameters: WEATHER_PARAMS }],
    });

    const done = events
      .filter((e) => e.type === "response.output_item.done" && e.item.type === "function_call")
      .map((e) => ({ call_id: e.item.call_id, arguments: e.item.arguments }));
    expect(done).toEqual([
      { call_id: "call_59014", arguments: '{"city":"Paris"}' },
      { call_id: "call_59015", arguments: '{"city":"Rome"}' },
    ]);
  });

  test("/v1/chat/completions gives each streamed call its own index", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }
    const events = await streamEvents("/v1/chat/completions", {
      model: MODEL,
      stream: true,
      messages: [{ role: "user", content: PROMPT }],
      tools: [{ type: "function", function: { name: "get_weather", parameters: WEATHER_PARAMS } }],
    });

    const calls = events.flatMap((e) =>
      (e.choices?.[0]?.delta?.tool_calls ?? []).map((tc: any) => ({
        index: tc.index,
        id: tc.id,
        arguments: tc.function.arguments,
      })),
    );
    expect(calls).toEqual([
      { index: 0, id: "call_59014", arguments: '{"city":"Paris"}' },
      { index: 1, id: "call_59015", arguments: '{"city":"Rome"}' },
    ]);
  });
});
