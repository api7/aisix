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

// E2E: an OpenAI-protocol client streaming from a Claude upstream receives the
// tool calls Claude makes. Claude streams a tool call as a `tool_use` content
// block — `content_block_start` carries its id and name, `input_json_delta`
// frames carry its arguments — and the client must see that as streamed
// `delta.tool_calls`: one index per call in the order they were made, the id,
// `type: "function"` and the name on a call's first fragment, the argument
// text after it, and `finish_reason: "tool_calls"`. Covered for the Anthropic
// Messages upstream, Claude on Vertex (`:streamRawPredict`, same event
// stream), and `/v1/responses`, which streams Claude through the same chunks.

const CALLER = "sk-anthropic-stream-tool-calls";
const CALLER_HASH = createHash("sha256").update(CALLER).digest("hex");

const WEATHER_ARGS = '{"city": "Paris", "unit": "celsius"}';

type Frame = Record<string, unknown>;

function messageStart(): Frame {
  return {
    type: "message_start",
    message: {
      id: "msg_stream_tools",
      type: "message",
      role: "assistant",
      content: [],
      model: "claude-sonnet-4-5",
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: 40, output_tokens: 1 },
    },
  };
}

function textBlock(index: number, text: string): Frame[] {
  return [
    { type: "content_block_start", index, content_block: { type: "text", text: "" } },
    { type: "content_block_delta", index, delta: { type: "text_delta", text } },
    { type: "content_block_stop", index },
  ];
}

function thinkingBlock(index: number, thinking: string): Frame[] {
  return [
    { type: "content_block_start", index, content_block: { type: "thinking", thinking: "", signature: "" } },
    { type: "content_block_delta", index, delta: { type: "thinking_delta", thinking } },
    { type: "content_block_delta", index, delta: { type: "signature_delta", signature: "SIGNATURE_BYTES" } },
    { type: "content_block_stop", index },
  ];
}

// Claude opens a tool_use block with an empty `input` and an empty first
// fragment, then streams the JSON in pieces.
function toolUseBlock(index: number, id: string, name: string, args: string, pieces: number): Frame[] {
  const size = Math.ceil(args.length / pieces);
  const fragments = [""];
  for (let i = 0; i < args.length; i += size) fragments.push(args.slice(i, i + size));
  return [
    { type: "content_block_start", index, content_block: { type: "tool_use", id, name, input: {} } },
    ...fragments.map((partial_json) => ({
      type: "content_block_delta",
      index,
      delta: { type: "input_json_delta", partial_json },
    })),
    { type: "content_block_stop", index },
  ];
}

function anthropicStream(blocks: Frame[]): string[] {
  return [
    messageStart(),
    ...blocks,
    { type: "message_delta", delta: { stop_reason: "tool_use", stop_sequence: null }, usage: { output_tokens: 57 } },
    { type: "message_stop" },
  ].map((f) => `event: ${f.type}\ndata: ${JSON.stringify(f)}\n\n`);
}

const STREAMS = {
  single: anthropicStream(toolUseBlock(0, "toolu_weather", "get_weather", WEATHER_ARGS, 3)),
  parallel: anthropicStream([
    ...toolUseBlock(0, "toolu_weather", "get_weather", WEATHER_ARGS, 2),
    // A tool without parameters: Claude streams no argument text for it.
    ...toolUseBlock(1, "toolu_time", "get_time", "", 0),
  ]),
  textThenTool: anthropicStream([
    ...textBlock(0, "Let me check the weather."),
    ...toolUseBlock(1, "toolu_weather", "get_weather", WEATHER_ARGS, 4),
  ]),
  thinkingThenTool: anthropicStream([
    ...thinkingBlock(0, "The user wants the weather in Paris."),
    ...toolUseBlock(1, "toolu_weather", "get_weather", WEATHER_ARGS, 2),
  ]),
} as const;

const TOOLS = [
  {
    type: "function",
    function: {
      name: "get_weather",
      parameters: { type: "object", properties: { city: { type: "string" }, unit: { type: "string" } } },
    },
  },
  { type: "function", function: { name: "get_time", parameters: { type: "object", properties: {} } } },
];

interface AssembledCall {
  id?: string;
  type?: string;
  name?: string;
  arguments: string;
  firstFragment: Record<string, any>;
}

function parseChatSse(raw: string): Array<Record<string, any>> {
  return raw
    .split("\n")
    .filter((l) => l.startsWith("data: ") && l !== "data: [DONE]")
    .map((l) => JSON.parse(l.slice(6)));
}

// Assembles streamed tool calls the way an OpenAI SDK client does: by index.
function assemble(chunks: Array<Record<string, any>>) {
  const calls: AssembledCall[] = [];
  let content = "";
  let reasoning = "";
  let finish: string | undefined;
  for (const chunk of chunks) {
    const choice = chunk.choices?.[0];
    if (!choice) continue;
    content += choice.delta?.content ?? "";
    reasoning += choice.delta?.reasoning_content ?? "";
    if (choice.finish_reason) finish = choice.finish_reason;
    for (const tc of choice.delta?.tool_calls ?? []) {
      const call = (calls[tc.index] ??= { arguments: "", firstFragment: tc });
      call.id ??= tc.id;
      call.type ??= tc.type;
      call.name ??= tc.function?.name;
      call.arguments += tc.function?.arguments ?? "";
    }
  }
  return { calls, content, reasoning, finish };
}

describe("Claude streaming tool calls reach OpenAI-protocol clients", () => {
  let app: SpawnedApp | undefined;
  const upstreams: Record<string, OpenAiUpstream> = {};
  let etcdReachable = false;

  const MODELS = {
    single: "stream-tools-single",
    parallel: "stream-tools-parallel",
    textThenTool: "stream-tools-text",
    thinkingThenTool: "stream-tools-thinking",
    vertex: "stream-tools-vertex-claude",
  } as const;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    for (const key of Object.keys(STREAMS) as Array<keyof typeof STREAMS>) {
      upstreams[key] = await startOpenAiUpstream({ rawStreamFrames: [...STREAMS[key]] });
    }
    upstreams.vertex = await startOpenAiUpstream({ rawStreamFrames: [...STREAMS.parallel] });

    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);
    for (const key of Object.keys(STREAMS) as Array<keyof typeof STREAMS>) {
      const pk = await seed.createProviderKey({
        display_name: `${MODELS[key]}-pk`,
        provider: "anthropic",
        adapter: "anthropic",
        secret: "sk-anth-mock",
        api_base: upstreams[key]!.baseUrl,
      });
      await seed.createModel({
        display_name: MODELS[key],
        provider: "anthropic",
        model_name: "claude-sonnet-4-5",
        provider_key_id: pk.id,
      });
    }
    const vertexPk = await seed.createProviderKey({
      display_name: `${MODELS.vertex}-pk`,
      provider: "google",
      adapter: "vertex",
      secret: JSON.stringify({ access_token: "ya29.stream-tools", project: "p", region: "us-east5" }),
      api_base: upstreams.vertex!.baseUrl,
    });
    await seed.createModel({
      display_name: MODELS.vertex,
      provider: "google",
      model_name: "claude-sonnet-4-5",
      provider_key_id: vertexPk.id,
    });
    await seed.createApiKey({ key_hash: CALLER_HASH, allowed_models: Object.values(MODELS) });
  });

  afterAll(async () => {
    await app?.exit();
    for (const u of Object.values(upstreams)) await u.close();
  });

  async function ready(): Promise<void> {
    const probe = new ProxyClient(app!.proxyUrl, CALLER);
    await waitConfigPropagation(async () => {
      const res = await probe.listModels();
      if (res.status !== 200) return false;
      const data = (res.body as { data?: Array<{ id?: string }> }).data ?? [];
      return Object.values(MODELS).every((m) => data.some((e) => e.id === m));
    });
  }

  async function streamChat(model: string) {
    const res = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { authorization: `Bearer ${CALLER}`, "content-type": "application/json" },
      body: JSON.stringify({
        model,
        stream: true,
        tools: TOOLS,
        messages: [{ role: "user", content: "What's the weather in Paris, and what time is it?" }],
      }),
    });
    expect(res.status).toBe(200);
    const raw = await res.text();
    return { raw, ...assemble(parseChatSse(raw)) };
  }

  function expectCallOpening(call: AssembledCall, id: string, name: string) {
    expect(call.firstFragment).toMatchObject({ id, type: "function", function: { name } });
    expect(call).toMatchObject({ id, type: "function", name });
  }

  test("a single tool call streams as tool_calls[0] with its arguments", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    await ready();
    const out = await streamChat(MODELS.single);
    expect(out.calls).toHaveLength(1);
    expectCallOpening(out.calls[0]!, "toolu_weather", "get_weather");
    expect(JSON.parse(out.calls[0]!.arguments)).toEqual({ city: "Paris", unit: "celsius" });
    expect(out.finish).toBe("tool_calls");
    expect(out.content).toBe("");
  });

  test("parallel tool calls stream as separate indices; a call without arguments gets {}", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    await ready();
    const out = await streamChat(MODELS.parallel);
    expect(out.calls).toHaveLength(2);
    expectCallOpening(out.calls[0]!, "toolu_weather", "get_weather");
    expectCallOpening(out.calls[1]!, "toolu_time", "get_time");
    expect(JSON.parse(out.calls[0]!.arguments)).toEqual({ city: "Paris", unit: "celsius" });
    expect(JSON.parse(out.calls[1]!.arguments)).toEqual({});
    expect(out.finish).toBe("tool_calls");
  });

  test("text before a tool call streams as content, and the call still opens at index 0", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    await ready();
    const out = await streamChat(MODELS.textThenTool);
    expect(out.content).toBe("Let me check the weather.");
    expect(out.calls).toHaveLength(1);
    expectCallOpening(out.calls[0]!, "toolu_weather", "get_weather");
    expect(JSON.parse(out.calls[0]!.arguments)).toEqual({ city: "Paris", unit: "celsius" });
    expect(out.finish).toBe("tool_calls");
  });

  test("thinking before a tool call streams as reasoning_content, and the call follows it", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    await ready();
    const out = await streamChat(MODELS.thinkingThenTool);
    expect(out.reasoning).toBe("The user wants the weather in Paris.");
    expect(out.calls).toHaveLength(1);
    expectCallOpening(out.calls[0]!, "toolu_weather", "get_weather");
    expect(JSON.parse(out.calls[0]!.arguments)).toEqual({ city: "Paris", unit: "celsius" });
    expect(out.finish).toBe("tool_calls");
    expect(out.raw).not.toContain("SIGNATURE_BYTES");
    const chunks = parseChatSse(out.raw);
    const firstReasoning = chunks.findIndex((c) => c.choices?.[0]?.delta?.reasoning_content);
    const firstCall = chunks.findIndex((c) => c.choices?.[0]?.delta?.tool_calls);
    expect(firstReasoning).toBeGreaterThanOrEqual(0);
    expect(firstReasoning).toBeLessThan(firstCall);
  });

  test("Claude on Vertex (streamRawPredict) streams parallel tool calls the same way", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    await ready();
    const out = await streamChat(MODELS.vertex);
    expect(out.calls).toHaveLength(2);
    expectCallOpening(out.calls[0]!, "toolu_weather", "get_weather");
    expectCallOpening(out.calls[1]!, "toolu_time", "get_time");
    expect(JSON.parse(out.calls[0]!.arguments)).toEqual({ city: "Paris", unit: "celsius" });
    expect(JSON.parse(out.calls[1]!.arguments)).toEqual({});
    expect(out.finish).toBe("tool_calls");
  });

  test("/v1/responses streaming: Claude's parallel tool calls become function_call items", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    await ready();
    const res = await fetch(`${app!.proxyUrl}/v1/responses`, {
      method: "POST",
      headers: { authorization: `Bearer ${CALLER}`, "content-type": "application/json" },
      body: JSON.stringify({
        model: MODELS.parallel,
        stream: true,
        input: "What's the weather in Paris, and what time is it?",
        tools: TOOLS.map((t) => ({ type: "function", ...t.function })),
      }),
    });
    expect(res.status).toBe(200);
    const events = parseChatSse(await res.text());
    const argDeltas = events.filter((e) => e.type === "response.function_call_arguments.delta");
    expect(argDeltas.length).toBeGreaterThan(0);
    const completed = events.find((e) => e.type === "response.completed");
    expect(completed).toBeDefined();
    const calls = (completed!.response.output as Array<Record<string, any>>).filter(
      (o) => o.type === "function_call",
    );
    expect(calls.map((c) => [c.call_id, c.name])).toEqual([
      ["toolu_weather", "get_weather"],
      ["toolu_time", "get_time"],
    ]);
    expect(JSON.parse(calls[0]!.arguments)).toEqual({ city: "Paris", unit: "celsius" });
    expect(JSON.parse(calls[1]!.arguments)).toEqual({});
  });
});
