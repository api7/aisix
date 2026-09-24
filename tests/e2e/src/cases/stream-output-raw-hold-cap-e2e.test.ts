import { createHash, randomUUID } from "node:crypto";
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

// E2E for the raw-byte bound on a held-back stream. `max_buffer_bytes`
// counts generated content only (#513), but the hold-back keeps whole frames,
// so each hold-back also bounds the raw bytes it keeps at 64 times the cap.
//
//  - A stream of frames that carry no content (pings, empty deltas,
//    keep-alives, base64 image previews) past that bound is a buffer-exceeded
//    event, honouring `on_buffer_exceeded`: fail_closed refuses, fail_open
//    releases unscanned.
//  - An ordinary token-by-token text stream still trips on the content cap:
//    content exactly at the cap is scanned and released, one byte more trips.
//
// Every stream ends with an email the guardrail masks whenever it scans, so a
// masked email proves nothing tripped and a raw one proves the stream went out
// unscanned.

const CALLER = "sk-stream-raw-hold-caller";
const CALLER_HASH = createHash("sha256").update(CALLER).digest("hex");
const CAP = 1_000;
// Comfortably past the raw bound (64 × CAP = 64 000 bytes).
const RAW_TARGET = 3 * 64 * CAP;
const EMAIL = "raw-probe@example.com";
const MASKED = "[EMAIL_REDACTED]";
const TAIL = `reach me at ${EMAIL}`;

const chatChunk = (delta: Record<string, unknown>, finish: string | null = null) =>
  JSON.stringify({
    id: "chatcmpl-raw",
    object: "chat.completion.chunk",
    model: "gpt-4o-mini",
    choices: [{ index: 0, delta, finish_reason: finish }],
  });

const repeatToBytes = (frame: string, bytes: number) =>
  Array.from({ length: Math.ceil(bytes / frame.length) }, () => frame);

// Chat: empty deltas, which the typed routes still hold as chunks.
const CHAT_CONTENT_FREE = [
  chatChunk({ role: "assistant" }),
  ...repeatToBytes(chatChunk({}), RAW_TARGET),
  chatChunk({ content: TAIL }),
  chatChunk({}, "stop"),
  "[DONE]",
];

// Bridged `/v1/responses`: the bridge holds the Responses events it encodes,
// and an empty delta encodes to none, so its content-free frames are tool
// calls with no arguments — each opens an output item.
const CHAT_EMPTY_TOOL_CALLS = [
  chatChunk({ role: "assistant" }),
  ...Array.from({ length: 1_000 }, (_, i) =>
    chatChunk({
      tool_calls: [{ index: i, id: `call_${i}`, type: "function", function: { name: "noop", arguments: "" } }],
    }),
  ),
  chatChunk({ content: TAIL }),
  chatChunk({}, "stop"),
  "[DONE]",
];

// Bridged `/v1/responses` whose upstream ends without a finish or usage
// chunk: the bridge closes the response itself. `response.created` and
// `response.in_progress` echo the request's `instructions`, together under the
// raw bound; the terminal event the bridge adds echoes them a third time and
// carries the stream past it.
const CHAT_NO_FINISH = [chatChunk({ role: "assistant" }), chatChunk({ content: TAIL })];
const LONG_INSTRUCTIONS = `Be brief. ${"i".repeat(25 * CAP)}`;

// Token-by-token text whose content totals `bytes`, ending with TAIL.
const tokens = (bytes: number) => {
  const filler = bytes - TAIL.length;
  return [
    ...Array.from({ length: Math.floor(filler / 4) }, () => "tok "),
    "x".repeat(filler % 4),
    TAIL,
  ].filter((t) => t.length > 0);
};

const chatText = (bytes: number) => [
  chatChunk({ role: "assistant" }),
  ...tokens(bytes).map((t) => chatChunk({ content: t })),
  chatChunk({}, "stop"),
  "[DONE]",
];

const anthropicFrame = (type: string, body: Record<string, unknown>) =>
  `event: ${type}\ndata: ${JSON.stringify({ type, ...body })}\n\n`;

const anthropicStream = (middle: string[], text: string[]) => [
  anthropicFrame("message_start", {
    message: {
      id: "msg_raw",
      type: "message",
      role: "assistant",
      content: [],
      model: "claude-3-5-haiku-20241022",
      stop_reason: null,
      usage: { input_tokens: 5, output_tokens: 1 },
    },
  }),
  ...middle,
  anthropicFrame("content_block_start", { index: 0, content_block: { type: "text", text: "" } }),
  ...text.map((t) =>
    anthropicFrame("content_block_delta", { index: 0, delta: { type: "text_delta", text: t } }),
  ),
  anthropicFrame("content_block_stop", { index: 0 }),
  anthropicFrame("message_delta", { delta: { stop_reason: "end_turn" }, usage: { output_tokens: 40 } }),
  anthropicFrame("message_stop", {}),
];

// Anthropic: `ping` events.
const ANTHROPIC_PINGS = anthropicStream(repeatToBytes(anthropicFrame("ping", {}), RAW_TARGET), [TAIL]);
const anthropicText = (bytes: number) => anthropicStream([], tokens(bytes));

// Responses: base64 image previews, which carry no generated text.
const B64 = "A".repeat(8_000);
const RESPONSES_PARTIAL_IMAGES = [
  JSON.stringify({
    type: "response.created",
    response: { id: "resp_raw", object: "response", status: "in_progress", model: "gpt-4o-mini", output: [] },
  }),
  ...Array.from({ length: Math.ceil(RAW_TARGET / B64.length) }, (_, i) =>
    JSON.stringify({
      type: "response.image_generation_call.partial_image",
      item_id: "ig_raw",
      output_index: 0,
      partial_image_index: i,
      partial_image_b64: B64,
    }),
  ),
  JSON.stringify({
    type: "response.output_text.delta",
    item_id: "msg_raw",
    output_index: 1,
    content_index: 0,
    delta: TAIL,
  }),
  JSON.stringify({
    type: "response.completed",
    response: {
      id: "resp_raw",
      object: "response",
      status: "completed",
      model: "gpt-4o-mini",
      output: [],
      usage: { input_tokens: 5, output_tokens: 40, total_tokens: 45 },
    },
  }),
];

// Passthrough: keep-alive chunks with no choices.
const PASSTHROUGH_KEEPALIVES = [
  ...repeatToBytes(JSON.stringify({ id: "chatcmpl-raw", object: "chat.completion.chunk", choices: [] }), RAW_TARGET),
  chatChunk({ content: TAIL }),
  chatChunk({}, "stop"),
  "[DONE]",
];

describe("held-back stream raw-byte bound", () => {
  let app: SpawnedApp | undefined;
  const upstreams: OpenAiUpstream[] = [];
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);

    const guardrail = async (name: string, onExceeded: string, action = "mask") =>
      seed.createGuardrail(
        {
          name,
          enabled: true,
          hook_point: "output",
          kind: "pii",
          detectors: [{ type: "email", action }],
          max_buffer_bytes: CAP,
          on_buffer_exceeded: onExceeded,
        },
        { attach: false },
      );
    const policies = {
      closed: await guardrail("raw-fail-closed", "fail_closed"),
      open: await guardrail("raw-fail-open", "fail_open"),
    };
    // A passthrough route relays bytes, so a mask can't show whether it
    // scanned; a block rule on the same email can.
    const routePolicies: Record<string, { id: string }> = {
      closed: await guardrail("raw-route-fail-closed", "fail_closed", "block"),
      open: await guardrail("raw-route-fail-open", "fail_open", "block"),
    };

    const upstream = async (events: string[], raw: boolean) => {
      const u = await startOpenAiUpstream(raw ? { rawStreamFrames: events } : { streamEvents: events });
      upstreams.push(u);
      return u;
    };
    const model = async (
      display: string,
      provider: "openai" | "anthropic",
      events: string[],
      guard: { id: string },
      opts: { raw?: boolean; pk?: Record<string, unknown> } = {},
    ) => {
      const u = await upstream(events, opts.raw ?? false);
      const pk = await seed.createProviderKey({
        display_name: `${display}-pk`,
        secret: "sk-mock",
        api_base: provider === "openai" ? `${u.baseUrl}/v1` : u.baseUrl,
        ...opts.pk,
      });
      const m = await seed.createModel({
        display_name: display,
        provider,
        model_name: provider === "openai" ? "gpt-4o-mini" : "claude-3-5-haiku-20241022",
        provider_key_id: pk.id,
      });
      await seed.attachGuardrailToModel(guard.id, m.id);
      return pk;
    };

    for (const [policy, guard] of Object.entries(policies)) {
      await model(`raw-chat-${policy}`, "openai", CHAT_CONTENT_FREE, guard);
      await model(`raw-msg-native-${policy}`, "anthropic", ANTHROPIC_PINGS, guard, { raw: true });
      await model(`raw-msg-bridge-${policy}`, "openai", CHAT_CONTENT_FREE, guard);
      await model(`raw-resp-native-${policy}`, "openai", RESPONSES_PARTIAL_IMAGES, guard);
      // `apis: {}`: no `/v1/responses` on this endpoint, so the route reaches
      // it through the chat bridge.
      await model(`raw-resp-bridge-${policy}`, "openai", CHAT_EMPTY_TOOL_CALLS, guard, { pk: { apis: {} } });
      await model(`raw-resp-bridge-eof-${policy}`, "openai", CHAT_NO_FINISH, guard, { pk: { apis: {} } });
      const backing = await model(`raw-route-backing-${policy}`, "openai", PASSTHROUGH_KEEPALIVES, guard);
      const route = await seed.createPassthroughRoute({
        name: `raw-route-${policy}`,
        path_prefix: `/passthrough/raw-${policy}`,
        target_url: String(backing.value.api_base),
        provider_key_id: backing.id,
      });
      await seed.update("guardrail_attachments", randomUUID(), {
        guardrail_id: routePolicies[policy]!.id,
        scope_type: "passthrough_route",
        scope_id: route.id,
        priority: 100,
      });
    }
    await model("raw-chat-at-cap", "openai", chatText(CAP), policies.closed);
    await model("raw-chat-over-cap", "openai", chatText(CAP + 1), policies.closed);
    await model("raw-msg-at-cap", "anthropic", anthropicText(CAP), policies.closed, { raw: true });
    await model("raw-msg-over-cap", "anthropic", anthropicText(CAP + 1), policies.closed, { raw: true });

    // Seeded last: this key authenticating implies the whole seed set landed.
    await seed.createApiKey({ key_hash: CALLER_HASH, allowed_models: ["*"], allowed_routes: ["*"] });
    const proxy = new ProxyClient(app.proxyUrl, CALLER);
    await waitConfigPropagation(async () => (await proxy.listModels()).status === 200);
  }, 90_000);

  afterAll(async () => {
    await app?.exit();
    await Promise.all(upstreams.map((u) => u.close()));
  });

  const post = async (path: string, body: Record<string, unknown>) => {
    const res = await fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${CALLER}`,
        "x-api-key": CALLER,
      },
      body: JSON.stringify({ ...body, stream: true }),
    });
    // A refusal before anything was sent is a 422 (native `/v1/responses`
    // holds the whole stream); the other routes refuse in-stream.
    expect([200, 422]).toContain(res.status);
    return res.text();
  };
  const chat = (model: string) => post("/v1/chat/completions", { model, messages: [{ role: "user", content: "go" }] });
  const messages = (model: string) =>
    post("/v1/messages", { model, max_tokens: 256, messages: [{ role: "user", content: "go" }] });
  const responses = (model: string) => post("/v1/responses", { model, input: "go" });
  const route = (policy: string) =>
    post(`/passthrough/raw-${policy}/chat/completions`, {
      model: "gpt-4o-mini",
      messages: [{ role: "user", content: "go" }],
    });

  const ready = (ctx: { skip: () => void }) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return false;
    }
    return true;
  };

  const surfaces: Array<[string, (policy: string) => Promise<string>]> = [
    ["/v1/chat/completions", (p) => chat(`raw-chat-${p}`)],
    ["/v1/messages (native)", (p) => messages(`raw-msg-native-${p}`)],
    ["/v1/messages (bridged)", (p) => messages(`raw-msg-bridge-${p}`)],
    ["/v1/responses (native)", (p) => responses(`raw-resp-native-${p}`)],
    ["/v1/responses (bridged)", (p) => responses(`raw-resp-bridge-${p}`)],
    [
      "/v1/responses (bridged, closed by the gateway)",
      (p) => post("/v1/responses", { model: `raw-resp-bridge-eof-${p}`, input: "go", instructions: LONG_INSTRUCTIONS }),
    ],
    ["passthrough route", (p) => route(p)],
  ];

  for (const [surface, send] of surfaces) {
    test(`${surface}: content-free frames past the raw bound fail closed`, async (ctx) => {
      if (!ready(ctx)) return;
      const body = await send("closed");
      expect(body).toContain("output_buffer_exceeded");
      expect(body).not.toContain(EMAIL);
      expect(body).not.toContain(MASKED);
    });

    test(`${surface}: content-free frames past the raw bound fail open unscanned`, async (ctx) => {
      if (!ready(ctx)) return;
      const body = await send("open");
      expect(body).not.toContain("output_buffer_exceeded");
      expect(body, "released unscanned, so the guardrail never ran").toContain(EMAIL);
      expect(body).not.toContain(MASKED);
      expect(body).not.toContain("content_filter");
    });
  }

  const textCases: Array<[string, (m: string) => Promise<string>, string]> = [
    ["/v1/chat/completions", chat, "raw-chat"],
    ["/v1/messages (native)", messages, "raw-msg"],
  ];
  for (const [surface, send, prefix] of textCases) {
    test(`${surface}: a token stream trips on the content cap, not the raw bound`, async (ctx) => {
      if (!ready(ctx)) return;
      const atCap = await send(`${prefix}-at-cap`);
      expect(atCap).not.toContain("output_buffer_exceeded");
      expect(atCap).toContain(MASKED);
      expect(atCap).not.toContain(EMAIL);
      const overCap = await send(`${prefix}-over-cap`);
      expect(overCap).toContain("output_buffer_exceeded");
      expect(overCap).not.toContain(EMAIL);
    });
  }
});
