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

// E2E for what an output guardrail's `max_buffer_bytes` measures while a
// streamed response is held back for inspection (#513), and for
// `on_buffer_exceeded` on every route that holds one back.
//
// The cap counts the model-generated content held — assistant text,
// reasoning, and tool-call arguments — and never the SSE/JSON framing
// around it, the same way on every route:
//
//  - a stream whose bulk is tool-call arguments, or reasoning, trips the cap
//    even though its assistant text alone stays well under it;
//  - a stream whose content stays under the cap is released and scanned as
//    before, however many frames (and so framing bytes) it takes to arrive.
//
// `on_buffer_exceeded: fail_open` releases what was held unscanned and
// streams the rest without an output scan — on `/v1/messages` and
// `/v1/responses` as on `/v1/chat/completions`. The observable is the mask:
// every stream ends with an email the guardrail masks whenever it scans, so
// a raw email proves the response went out unscanned, and a masked one
// proves the cap never tripped.

const CALLER = "sk-stream-buffer-cap-caller";
const CALLER_HASH = createHash("sha256").update(CALLER).digest("hex");
const CAP = 1_000;
const EMAIL = "cap-probe@example.com";
const MASKED = "[EMAIL_REDACTED]";
const TAIL = `reach me at ${EMAIL}`;
// 30 pieces of 100 bytes: three times the cap once counted.
const PIECES = Array.from({ length: 30 }, (_, i) => `${String(i).padStart(2, "0")}${"r".repeat(98)}`);

const chatChunk = (delta: Record<string, unknown>, finish: string | null = null) =>
  JSON.stringify({
    id: "chatcmpl-cap",
    object: "chat.completion.chunk",
    model: "gpt-4o-mini",
    choices: [{ index: 0, delta, finish_reason: finish }],
  });

const CHAT_TOOL_HEAVY = [
  chatChunk({ role: "assistant" }),
  chatChunk({
    tool_calls: [
      { index: 0, id: "call_1", type: "function", function: { name: "record", arguments: "" } },
    ],
  }),
  ...PIECES.map((p) => chatChunk({ tool_calls: [{ index: 0, function: { arguments: p } }] })),
  chatChunk({}, "tool_calls"),
  "[DONE]",
];

const CHAT_REASONING_HEAVY = [
  chatChunk({ role: "assistant" }),
  ...PIECES.map((p) => chatChunk({ reasoning_content: p })),
  chatChunk({ content: TAIL }),
  chatChunk({}, "stop"),
  "[DONE]",
];

// ~600 bytes of text over 60 frames: the frames' own bytes exceed the cap,
// the text they carry does not.
const TEXT_SLICES = Array.from({ length: 60 }, () => "0123456789");
const UNDER_CAP_TEXT = TEXT_SLICES.join("");

const CHAT_TEXT_UNDER_CAP = [
  chatChunk({ role: "assistant" }),
  ...TEXT_SLICES.map((s) => chatChunk({ content: s })),
  chatChunk({ content: TAIL }),
  chatChunk({}, "stop"),
  "[DONE]",
];

const anthropicStream = (blocks: Array<{ kind: "thinking" | "text"; deltas: string[] }>) => [
  JSON.stringify({
    type: "message_start",
    message: {
      id: "msg_cap",
      type: "message",
      role: "assistant",
      content: [],
      model: "claude-3-5-haiku-20241022",
      stop_reason: null,
      usage: { input_tokens: 5, output_tokens: 1 },
    },
  }),
  ...blocks.flatMap((b, index) => [
    JSON.stringify({
      type: "content_block_start",
      index,
      content_block: b.kind === "thinking" ? { type: "thinking", thinking: "" } : { type: "text", text: "" },
    }),
    ...b.deltas.map((d) =>
      JSON.stringify({
        type: "content_block_delta",
        index,
        delta: b.kind === "thinking" ? { type: "thinking_delta", thinking: d } : { type: "text_delta", text: d },
      }),
    ),
    JSON.stringify({ type: "content_block_stop", index }),
  ]),
  JSON.stringify({ type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 40 } }),
  JSON.stringify({ type: "message_stop" }),
];

const ANTHROPIC_THINKING_HEAVY = anthropicStream([
  { kind: "thinking", deltas: PIECES },
  { kind: "text", deltas: [TAIL] },
]);
const ANTHROPIC_TEXT_UNDER_CAP = anthropicStream([{ kind: "text", deltas: [...TEXT_SLICES, TAIL] }]);

const RESPONSES_REASONING_HEAVY = [
  JSON.stringify({
    type: "response.created",
    response: { id: "resp_cap", object: "response", status: "in_progress", model: "gpt-4o-mini", output: [] },
  }),
  ...PIECES.map((p) =>
    JSON.stringify({
      type: "response.reasoning_summary_text.delta",
      item_id: "rs_cap",
      output_index: 0,
      summary_index: 0,
      delta: p,
    }),
  ),
  JSON.stringify({
    type: "response.output_text.delta",
    item_id: "msg_cap",
    output_index: 1,
    content_index: 0,
    delta: TAIL,
  }),
  JSON.stringify({
    type: "response.completed",
    response: {
      id: "resp_cap",
      object: "response",
      status: "completed",
      model: "gpt-4o-mini",
      output: [
        {
          type: "message",
          id: "msg_cap",
          role: "assistant",
          status: "completed",
          content: [{ type: "output_text", text: TAIL, annotations: [] }],
        },
      ],
      usage: { input_tokens: 5, output_tokens: 40, total_tokens: 45 },
    },
  }),
];

describe("streamed output hold-back cap measures held content (#513)", () => {
  let app: SpawnedApp | undefined;
  const upstreams: OpenAiUpstream[] = [];
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);

    const guardrail = async (name: string, onExceeded: string) =>
      seed.createGuardrail(
        {
          name,
          enabled: true,
          hook_point: "output",
          kind: "pii",
          detectors: [{ type: "email", action: "mask" }],
          max_buffer_bytes: CAP,
          on_buffer_exceeded: onExceeded,
        },
        { attach: false },
      );
    const failClosed = await guardrail("cap-fail-closed", "fail_closed");
    const failOpen = await guardrail("cap-fail-open", "fail_open");

    const model = async (
      display: string,
      provider: "openai" | "anthropic",
      streamEvents: string[],
      guard: { id: string },
      pkExtra: Record<string, unknown> = {},
    ) => {
      const upstream = await startOpenAiUpstream({ streamEvents });
      upstreams.push(upstream);
      const pk = await seed.createProviderKey({
        display_name: `${display}-pk`,
        secret: "sk-mock",
        api_base: provider === "openai" ? `${upstream.baseUrl}/v1` : upstream.baseUrl,
        ...pkExtra,
      });
      const m = await seed.createModel({
        display_name: display,
        provider,
        model_name: provider === "openai" ? "gpt-4o-mini" : "claude-3-5-haiku-20241022",
        provider_key_id: pk.id,
      });
      await seed.attachGuardrailToModel(guard.id, m.id);
    };

    await model("cap-chat-tools", "openai", CHAT_TOOL_HEAVY, failClosed);
    await model("cap-chat-reasoning", "openai", CHAT_REASONING_HEAVY, failClosed);
    await model("cap-chat-text", "openai", CHAT_TEXT_UNDER_CAP, failClosed);
    await model("cap-msg-native", "anthropic", ANTHROPIC_THINKING_HEAVY, failOpen);
    await model("cap-msg-native-text", "anthropic", ANTHROPIC_TEXT_UNDER_CAP, failOpen);
    await model("cap-msg-bridge", "openai", CHAT_REASONING_HEAVY, failOpen);
    await model("cap-resp-native", "openai", RESPONSES_REASONING_HEAVY, failOpen);
    // `apis: {}`: an OpenAI-compatible endpoint with no `/v1/responses`, so
    // `/v1/responses` reaches it through the chat bridge.
    await model("cap-resp-bridge", "openai", CHAT_REASONING_HEAVY, failOpen, { apis: {} });

    // Seeded last: this key authenticating implies the whole seed set landed.
    await seed.createApiKey({
      key_hash: CALLER_HASH,
      allowed_models: [
        "cap-chat-tools",
        "cap-chat-reasoning",
        "cap-chat-text",
        "cap-msg-native",
        "cap-msg-native-text",
        "cap-msg-bridge",
        "cap-resp-native",
        "cap-resp-bridge",
      ],
    });
    const proxy = new ProxyClient(app.proxyUrl, CALLER);
    await waitConfigPropagation(async () => (await proxy.listModels()).status === 200);
  });

  afterAll(async () => {
    await app?.exit();
    await Promise.all(upstreams.map((u) => u.close()));
  });

  const post = (path: string, body: Record<string, unknown>) =>
    fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${CALLER}`,
        "x-api-key": CALLER,
      },
      body: JSON.stringify({ ...body, stream: true }),
    });
  const chat = (model: string) => post("/v1/chat/completions", { model, messages: [{ role: "user", content: "go" }] });
  const messages = (model: string) =>
    post("/v1/messages", { model, max_tokens: 256, messages: [{ role: "user", content: "go" }] });
  const responses = (model: string) => post("/v1/responses", { model, input: "go" });

  const ready = (ctx: { skip: () => void }) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return false;
    }
    return true;
  };

  test("chat: a stream of tool-call arguments over the cap fails closed", async (ctx) => {
    if (!ready(ctx)) return;
    const res = await chat("cap-chat-tools");
    expect(res.status).toBe(200);
    const body = await res.text();
    expect(body).toContain("output_buffer_exceeded");
    // Nothing held was released: not even the first argument piece.
    expect(body).not.toContain(PIECES[0]);
  });

  test("chat: a stream of reasoning over the cap fails closed", async (ctx) => {
    if (!ready(ctx)) return;
    const res = await chat("cap-chat-reasoning");
    expect(res.status).toBe(200);
    const body = await res.text();
    expect(body).toContain("output_buffer_exceeded");
    expect(body).not.toContain(PIECES[0]);
    expect(body).not.toContain(EMAIL);
  });

  test("chat: a text stream under the cap is scanned and released whole", async (ctx) => {
    if (!ready(ctx)) return;
    const res = await chat("cap-chat-text");
    expect(res.status).toBe(200);
    const body = await res.text();
    expect(body).not.toContain("output_buffer_exceeded");
    expect(body).toContain(MASKED);
    expect(body).not.toContain(EMAIL);
    const text = body
      .split("\n")
      .filter((l) => l.startsWith("data: {"))
      .map((l) => (JSON.parse(l.slice(6)) as { choices: Array<{ delta: { content?: string } }> }).choices[0]?.delta.content ?? "")
      .join("");
    expect(text.startsWith(UNDER_CAP_TEXT)).toBe(true);
  });

  test("messages (native): fail_open releases a thinking-heavy stream unscanned", async (ctx) => {
    if (!ready(ctx)) return;
    const res = await messages("cap-msg-native");
    expect(res.status).toBe(200);
    const body = await res.text();
    expect(body).not.toContain("event: error");
    expect(body).toContain(PIECES[29]);
    expect(body, "released unscanned, so the mask never ran").toContain(EMAIL);
    expect(body).toContain("message_stop");
  });

  test("messages (native): text under the cap is scanned however many frames carry it", async (ctx) => {
    if (!ready(ctx)) return;
    const res = await messages("cap-msg-native-text");
    expect(res.status).toBe(200);
    const body = await res.text();
    expect(body).not.toContain("event: error");
    expect(body).toContain(MASKED);
    expect(body).not.toContain(EMAIL);
  });

  test("messages (bridged): fail_open releases a reasoning-heavy stream unscanned", async (ctx) => {
    if (!ready(ctx)) return;
    const res = await messages("cap-msg-bridge");
    expect(res.status).toBe(200);
    const body = await res.text();
    expect(body).not.toContain("event: error");
    expect(body, "released unscanned, so the mask never ran").toContain(EMAIL);
    expect(body).toContain("message_stop");
  });

  test("responses (native): fail_open releases a reasoning-heavy stream unscanned", async (ctx) => {
    if (!ready(ctx)) return;
    const res = await responses("cap-resp-native");
    expect(res.status).toBe(200);
    const body = await res.text();
    expect(body).not.toContain("output_buffer_exceeded");
    expect(body).toContain(PIECES[29]);
    expect(body, "released unscanned, so the mask never ran").toContain(EMAIL);
    expect(body).toContain("response.completed");
  });

  test("responses (bridged): fail_open releases a reasoning-heavy stream unscanned", async (ctx) => {
    if (!ready(ctx)) return;
    const res = await responses("cap-resp-bridge");
    expect(res.status).toBe(200);
    const body = await res.text();
    expect(body).not.toContain("output_buffer_exceeded");
    expect(body).toContain(PIECES[29]);
    expect(body, "released unscanned, so the mask never ran").toContain(EMAIL);
    expect(body).toContain("response.completed");
  });
});
