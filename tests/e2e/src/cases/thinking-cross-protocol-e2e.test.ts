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

// E2E: a model's chain-of-thought survives a protocol translation in both
// directions (AISIX-Cloud#1784).
//
// 1. `/v1/messages` (Anthropic clients such as Claude Code) dispatched onto an
//    OpenAI-compatible reasoning upstream (DeepSeek-style
//    `reasoning_content`): the reasoning reaches the client as an Anthropic
//    `thinking` content block — first in the content, ahead of text and
//    tool_use — and, when the client replays that block in the next turn's
//    history, it reaches the upstream again as the assistant turn's
//    `reasoning_content`. Such upstreams require prior turns' reasoning back
//    on every tool-using turn, so the replay is what keeps a multi-turn agent
//    session working once the thinking reaches the client.
//
// 2. `/v1/chat/completions` dispatched onto an Anthropic upstream: Anthropic
//    `thinking` content blocks / `thinking_delta` events reach the client as
//    `reasoning_content`, the field OpenAI-compatible clients read reasoning
//    from.
//
// The upstream issues no signature for reasoning it reports as
// `reasoning_content`, so the Anthropic block carries `signature: ""`.

const CALLER = "sk-thinking-cross-protocol";
const CALLER_HASH = createHash("sha256").update(CALLER).digest("hex");

const REASONING_A = "The user wants a greeting. ";
const REASONING_B = "Keep it short.";
const ANSWER = "Hello!";

function chatCompletion(message: Record<string, unknown>, finish: string) {
  return {
    id: "chatcmpl-think",
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model: "deepseek-v4-pro",
    choices: [{ index: 0, message: { role: "assistant", ...message }, finish_reason: finish }],
    usage: { prompt_tokens: 12, completion_tokens: 20, total_tokens: 32 },
  };
}

function chatChunk(delta: Record<string, unknown>, finish: string | null = null) {
  return JSON.stringify({
    id: "chatcmpl-think-s",
    object: "chat.completion.chunk",
    model: "deepseek-v4-pro",
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
}

// The first chunk carries nothing but reasoning — no role, no content.
const OPENAI_STREAM = [
  chatChunk({ reasoning_content: REASONING_A }),
  chatChunk({ reasoning_content: REASONING_B }),
  chatChunk({ content: ANSWER }),
  chatChunk({}, "stop"),
  JSON.stringify({
    id: "chatcmpl-think-s",
    object: "chat.completion.chunk",
    model: "deepseek-v4-pro",
    choices: [],
    usage: { prompt_tokens: 12, completion_tokens: 20, total_tokens: 32 },
  }),
  "[DONE]",
];

const ANTHROPIC_NON_STREAM = {
  id: "msg_think",
  type: "message",
  role: "assistant",
  model: "claude-sonnet-4-5",
  content: [
    { type: "thinking", thinking: "First idea.", signature: "sig-a" },
    { type: "redacted_thinking", data: "REDACTED_CIPHERTEXT" },
    { type: "thinking", thinking: "Second idea.", signature: "sig-b" },
    { type: "text", text: ANSWER },
  ],
  stop_reason: "end_turn",
  usage: { input_tokens: 10, output_tokens: 30 },
};

const ANTHROPIC_STREAM = [
  JSON.stringify({
    type: "message_start",
    message: {
      id: "msg_think_s",
      type: "message",
      role: "assistant",
      content: [],
      model: "claude-sonnet-4-5",
      stop_reason: null,
      usage: { input_tokens: 10, output_tokens: 1 },
    },
  }),
  JSON.stringify({
    type: "content_block_start",
    index: 0,
    content_block: { type: "thinking", thinking: "", signature: "" },
  }),
  JSON.stringify({
    type: "content_block_delta",
    index: 0,
    delta: { type: "thinking_delta", thinking: "Streamed " },
  }),
  JSON.stringify({
    type: "content_block_delta",
    index: 0,
    delta: { type: "thinking_delta", thinking: "thought." },
  }),
  JSON.stringify({
    type: "content_block_delta",
    index: 0,
    delta: { type: "signature_delta", signature: "SIGNATURE_BYTES" },
  }),
  JSON.stringify({ type: "content_block_stop", index: 0 }),
  JSON.stringify({
    type: "content_block_start",
    index: 1,
    content_block: { type: "text", text: "" },
  }),
  JSON.stringify({
    type: "content_block_delta",
    index: 1,
    delta: { type: "text_delta", text: ANSWER },
  }),
  JSON.stringify({ type: "content_block_stop", index: 1 }),
  JSON.stringify({
    type: "message_delta",
    delta: { stop_reason: "end_turn" },
    usage: { output_tokens: 30 },
  }),
  JSON.stringify({ type: "message_stop" }),
];

interface SseEvent {
  event: string;
  data: Record<string, any>;
}

function parseAnthropicSse(raw: string): SseEvent[] {
  const events: SseEvent[] = [];
  for (const frame of raw.split("\n\n")) {
    const lines = frame.split("\n");
    const event = lines.find((l) => l.startsWith("event: "))?.slice(7);
    const data = lines.find((l) => l.startsWith("data: "))?.slice(6);
    if (!event || !data) continue;
    events.push({ event, data: JSON.parse(data) });
  }
  return events;
}

function parseChatSse(raw: string): Array<Record<string, any>> {
  return raw
    .split("\n")
    .filter((l) => l.startsWith("data: ") && l !== "data: [DONE]")
    .map((l) => JSON.parse(l.slice(6)));
}

describe("thinking ↔ reasoning_content across protocols (AISIX-Cloud#1784)", () => {
  let app: SpawnedApp | undefined;
  const upstreams: Record<string, OpenAiUpstream> = {};
  let etcdReachable = false;

  const MODELS = {
    oaiText: "think-oai-text",
    oaiTool: "think-oai-tool",
    oaiStream: "think-oai-stream",
    anthNonStream: "think-anth-nonstream",
    anthStream: "think-anth-stream",
  } as const;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstreams.oaiText = await startOpenAiUpstream({
      nonStreamBody: chatCompletion(
        { content: ANSWER, reasoning_content: REASONING_A + REASONING_B },
        "stop",
      ),
    });
    upstreams.oaiTool = await startOpenAiUpstream({
      nonStreamBody: chatCompletion(
        {
          content: null,
          reasoning_content: REASONING_A,
          tool_calls: [
            {
              id: "call_1",
              type: "function",
              function: { name: "get_time", arguments: '{"tz":"UTC"}' },
            },
          ],
        },
        "tool_calls",
      ),
    });
    upstreams.oaiStream = await startOpenAiUpstream({ streamEvents: OPENAI_STREAM });
    upstreams.anthNonStream = await startOpenAiUpstream({ nonStreamBody: ANTHROPIC_NON_STREAM });
    upstreams.anthStream = await startOpenAiUpstream({ streamEvents: ANTHROPIC_STREAM });

    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);
    for (const key of ["oaiText", "oaiTool", "oaiStream"] as const) {
      const pk = await seed.createProviderKey({
        display_name: `${MODELS[key]}-pk`,
        secret: "sk-mock",
        api_base: `${upstreams[key]!.baseUrl}/v1`,
      });
      await seed.createModel({
        display_name: MODELS[key],
        provider: "openai",
        model_name: "deepseek-v4-pro",
        provider_key_id: pk.id,
      });
    }
    for (const key of ["anthNonStream", "anthStream"] as const) {
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

  function post(path: string, body: unknown): Promise<Response> {
    return fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: { "x-api-key": CALLER, authorization: `Bearer ${CALLER}`, "content-type": "application/json" },
      body: JSON.stringify(body),
    });
  }

  test("/v1/messages non-streaming: reasoning becomes a leading thinking block before text", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    await ready();
    const res = await post("/v1/messages", {
      model: MODELS.oaiText,
      max_tokens: 256,
      messages: [{ role: "user", content: "say hi" }],
    });
    expect(res.status).toBe(200);
    const body = (await res.json()) as { content: unknown[] };
    expect(body.content).toEqual([
      { type: "thinking", thinking: REASONING_A + REASONING_B, signature: "" },
      { type: "text", text: ANSWER },
    ]);
  });

  test("/v1/messages non-streaming: thinking block precedes tool_use", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    await ready();
    const res = await post("/v1/messages", {
      model: MODELS.oaiTool,
      max_tokens: 256,
      tools: [{ name: "get_time", input_schema: { type: "object", properties: {} } }],
      messages: [{ role: "user", content: "what time is it" }],
    });
    expect(res.status).toBe(200);
    const body = (await res.json()) as { content: Array<Record<string, unknown>>; stop_reason: string };
    expect(body.stop_reason).toBe("tool_use");
    expect(body.content.map((b) => b.type)).toEqual(["thinking", "tool_use"]);
    expect(body.content[0]).toEqual({ type: "thinking", thinking: REASONING_A, signature: "" });
    expect(body.content[1]).toMatchObject({ id: "call_1", name: "get_time", input: { tz: "UTC" } });
  });

  test("/v1/messages streaming: a reasoning-first stream opens with message_start, then a thinking block, then text", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    await ready();
    const res = await post("/v1/messages", {
      model: MODELS.oaiStream,
      max_tokens: 256,
      stream: true,
      messages: [{ role: "user", content: "say hi" }],
    });
    expect(res.status).toBe(200);
    const events = parseAnthropicSse(await res.text());
    const summary = events.map((e) => {
      switch (e.event) {
        case "content_block_start":
          return `start ${e.data.index} ${e.data.content_block.type}`;
        case "content_block_delta":
          return `delta ${e.data.index} ${e.data.delta.type}`;
        case "content_block_stop":
          return `stop ${e.data.index}`;
        default:
          return e.event;
      }
    });
    expect(summary).toEqual([
      "message_start",
      "start 0 thinking",
      "delta 0 thinking_delta",
      "delta 0 thinking_delta",
      "stop 0",
      "start 1 text",
      "delta 1 text_delta",
      "stop 1",
      "message_delta",
      "message_stop",
    ]);
    const start = events.find((e) => e.event === "content_block_start")!;
    expect(start.data.content_block).toEqual({ type: "thinking", thinking: "", signature: "" });
    const thinking = events
      .filter((e) => e.data.delta?.type === "thinking_delta")
      .map((e) => e.data.delta.thinking)
      .join("");
    expect(thinking).toBe(REASONING_A + REASONING_B);
    const text = events.find((e) => e.data.delta?.type === "text_delta")!;
    expect(text.data.delta.text).toBe(ANSWER);
  });

  test("/v1/messages history: assistant thinking replays upstream as reasoning_content", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    await ready();
    const upstream = upstreams.oaiTool!;
    const baseline = upstream.receivedRequests.length;
    const res = await post("/v1/messages", {
      model: MODELS.oaiTool,
      max_tokens: 256,
      tools: [{ name: "get_time", input_schema: { type: "object", properties: {} } }],
      messages: [
        { role: "user", content: "what time is it" },
        {
          role: "assistant",
          content: [
            { type: "thinking", thinking: "I need the clock.", signature: "" },
            { type: "redacted_thinking", data: "REDACTED_CIPHERTEXT" },
            { type: "thinking", thinking: "Call get_time.", signature: "sig-x" },
            { type: "tool_use", id: "call_0", name: "get_time", input: {} },
          ],
        },
        {
          role: "user",
          content: [{ type: "tool_result", tool_use_id: "call_0", content: "12:00" }],
        },
      ],
    });
    expect(res.status).toBe(200);
    const sent = upstream.receivedRequests
      .slice(baseline)
      .find((r) => r.path === "/v1/chat/completions");
    expect(sent).toBeDefined();
    const messages = (JSON.parse(sent!.body) as { messages: Array<Record<string, unknown>> }).messages;
    const assistant = messages.find((m) => m.role === "assistant")!;
    expect(assistant.reasoning_content).toBe("I need the clock.\nCall get_time.");
    expect(assistant.tool_calls).toMatchObject([{ id: "call_0", function: { name: "get_time" } }]);
    expect(assistant).not.toHaveProperty("thinking_blocks");
    expect(sent!.body).not.toContain("REDACTED_CIPHERTEXT");
    expect(sent!.body).not.toContain("sig-x");
  });

  test("/v1/chat/completions → Anthropic non-streaming: thinking blocks surface as reasoning_content", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    await ready();
    const res = await post("/v1/chat/completions", {
      model: MODELS.anthNonStream,
      messages: [{ role: "user", content: "say hi" }],
    });
    expect(res.status).toBe(200);
    const raw = await res.text();
    const message = (JSON.parse(raw) as { choices: Array<{ message: Record<string, unknown> }> })
      .choices[0]!.message;
    expect(message.content).toBe(ANSWER);
    expect(message.reasoning_content).toBe("First idea.\nSecond idea.");
    expect(raw).not.toContain("REDACTED_CIPHERTEXT");
    expect(raw).not.toContain("sig-a");
  });

  test("/v1/chat/completions → Anthropic streaming: thinking_delta surfaces as delta.reasoning_content", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    await ready();
    const res = await post("/v1/chat/completions", {
      model: MODELS.anthStream,
      stream: true,
      messages: [{ role: "user", content: "say hi" }],
    });
    expect(res.status).toBe(200);
    const raw = await res.text();
    const chunks = parseChatSse(raw);
    const reasoning = chunks
      .map((c) => c.choices?.[0]?.delta?.reasoning_content ?? "")
      .join("");
    const content = chunks.map((c) => c.choices?.[0]?.delta?.content ?? "").join("");
    expect(reasoning).toBe("Streamed thought.");
    expect(content).toBe(ANSWER);
    // Reasoning precedes the answer, as the upstream produced it.
    const firstReasoning = chunks.findIndex((c) => c.choices?.[0]?.delta?.reasoning_content);
    const firstContent = chunks.findIndex((c) => c.choices?.[0]?.delta?.content);
    expect(firstReasoning).toBeGreaterThanOrEqual(0);
    expect(firstReasoning).toBeLessThan(firstContent);
    expect(raw).not.toContain("SIGNATURE_BYTES");
  });
});
