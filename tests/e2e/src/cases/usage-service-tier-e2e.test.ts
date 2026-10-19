import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
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

// E2E: the usage event records the service tier the UPSTREAM RESPONSE says
// it served the request on (`service_tier`, exported to OTLP as
// `aisix.service_tier`), verbatim and never derived from the request:
//
// - OpenAI-shape upstreams: the response's top-level `service_tier`, on
//   chat completions (non-streamed and streamed) and native Responses.
// - Anthropic `/v1/messages`: `usage.speed` when it is not "standard",
//   otherwise `usage.service_tier`.
// - An upstream that reports nothing leaves the field absent.
//
// And the client-visible bodies report the served tier, not the requested
// one: a chat completion carries the OpenAI-shape upstream's own
// `service_tier`, and a Responses object bridged onto a chat upstream
// reports that upstream's tier, or "default" when it sent none.
//
// References:
// - https://platform.openai.com/docs/api-reference/chat/object#chat/object-service_tier
// - https://platform.openai.com/docs/api-reference/responses/object#responses/object-service_tier
// - https://platform.claude.com/docs/en/api/service-tiers
// - https://platform.claude.com/docs/en/build-with-claude/fast-mode

const CALLER = "sk-usage-service-tier-caller";
const CALLER_HASH = createHash("sha256").update(CALLER).digest("hex");

const CREDENTIAL_REF = "e2e";
const SLS_PROJECT = "service-tier-proj";
const LOGSTORE = "service-tier-store";

const CHAT = "tier-chat";
const CHAT_STREAM = "tier-chat-stream";
const CHAT_NONE = "tier-chat-none";
const RESP = "tier-responses";
const RESP_STREAM = "tier-responses-stream";
const MSG_FAST = "tier-messages-fast";
const MSG_TIER = "tier-messages-tier";
const MSG_STREAM = "tier-messages-stream";
const BRIDGE = "tier-bridge";
const BRIDGE_NONE = "tier-bridge-none";
const BRIDGE_STREAM = "tier-bridge-stream";

function chatBody(tier?: string): Record<string, unknown> {
  return {
    id: "chatcmpl-tier",
    object: "chat.completion",
    created: 1_700_000_000,
    model: "gpt-4o-mini",
    choices: [{ index: 0, message: { role: "assistant", content: "ok" }, finish_reason: "stop" }],
    usage: { prompt_tokens: 3, completion_tokens: 1, total_tokens: 4 },
    ...(tier === undefined ? {} : { service_tier: tier }),
  };
}

function chatStream(tier: string): string[] {
  const chunk = (json: Record<string, unknown>) =>
    JSON.stringify({
      id: "chatcmpl-tier-stream",
      object: "chat.completion.chunk",
      created: 1_700_000_000,
      model: "gpt-4o-mini",
      service_tier: tier,
      ...json,
    });
  return [
    chunk({ choices: [{ index: 0, delta: { role: "assistant", content: "hi" }, finish_reason: null }] }),
    chunk({ choices: [{ index: 0, delta: {}, finish_reason: "stop" }] }),
    chunk({ choices: [], usage: { prompt_tokens: 2, completion_tokens: 1, total_tokens: 3 } }),
    "[DONE]",
  ];
}

function responsesObject(tier: string, status: string, withUsage: boolean): Record<string, unknown> {
  return {
    id: "resp_tier",
    object: "response",
    created_at: 1_700_000_000,
    status,
    model: "gpt-4o-mini",
    service_tier: tier,
    output: withUsage
      ? [
          {
            type: "message",
            id: "msg_tier",
            status: "completed",
            role: "assistant",
            content: [{ type: "output_text", text: "ok", annotations: [] }],
          },
        ]
      : [],
    usage: withUsage ? { input_tokens: 3, output_tokens: 1, total_tokens: 4 } : null,
  };
}

const RESPONSES_STREAM = [
  // The snapshot at creation names the requested mode; the terminal one
  // names the tier that actually served — the last one wins.
  JSON.stringify({ type: "response.created", response: responsesObject("auto", "in_progress", false) }),
  JSON.stringify({ type: "response.completed", response: responsesObject("flex", "completed", true) }),
];

function anthropicBody(usage: Record<string, unknown>): Record<string, unknown> {
  return {
    id: "msg_tier",
    type: "message",
    role: "assistant",
    model: "claude-3-5-haiku-20241022",
    content: [{ type: "text", text: "ok" }],
    stop_reason: "end_turn",
    usage: { input_tokens: 3, output_tokens: 1, ...usage },
  };
}

const ANTHROPIC_STREAM = [
  `event: message_start\ndata: ${JSON.stringify({
    type: "message_start",
    message: {
      id: "msg_tier_stream",
      role: "assistant",
      content: [],
      model: "claude-3-5-haiku-20241022",
      stop_reason: null,
      usage: { input_tokens: 3, output_tokens: 1, service_tier: "standard", speed: "fast" },
    },
  })}\n\n`,
  `event: content_block_delta\ndata: ${JSON.stringify({
    type: "content_block_delta",
    index: 0,
    delta: { type: "text_delta", text: "ok" },
  })}\n\n`,
  `event: message_delta\ndata: ${JSON.stringify({
    type: "message_delta",
    delta: { stop_reason: "end_turn" },
    usage: { output_tokens: 2 },
  })}\n\n`,
  `event: message_stop\ndata: ${JSON.stringify({ type: "message_stop" })}\n\n`,
];

/** Every `data:` payload of an SSE body, parsed. */
function sseFrames(text: string): Record<string, unknown>[] {
  return text
    .split("\n")
    .filter((l) => l.startsWith("data: ") && l.slice(6).trim() !== "[DONE]")
    .map((l) => JSON.parse(l.slice(6)) as Record<string, unknown>);
}

describe("usage events record the service tier the upstream served on", () => {
  let etcdReachable = false;
  let app: SpawnedApp | undefined;
  let sls: MockSls | undefined;
  let otlp: MockOtlp | undefined;
  const upstreams: OpenAiUpstream[] = [];

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    sls = await startMockSls();
    otlp = await startMockOtlp();
    app = await spawnApp({
      extraEnv: {
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: "mock-ak-id",
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: "mock-ak-secret",
      },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createObservabilityExporter({
      name: "service-tier-sls",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: SLS_PROJECT,
      logstore: LOGSTORE,
      credential_ref: CREDENTIAL_REF,
    });
    await seed.createObservabilityExporter({
      name: "service-tier-otlp",
      enabled: true,
      kind: "otlp_http",
      endpoint: otlp.url,
    });

    const model = async (
      name: string,
      provider: "openai" | "deepseek" | "anthropic",
      opts: Parameters<typeof startOpenAiUpstream>[0],
    ) => {
      const upstream = await startOpenAiUpstream(opts);
      upstreams.push(upstream);
      const anthropic = provider === "anthropic";
      const pk = await seed.createProviderKey({
        display_name: `${name}-pk`,
        secret: "sk-mock",
        // The Anthropic bridge appends `/v1/messages` to the bare host.
        api_base: anthropic ? upstream.baseUrl : `${upstream.baseUrl}/v1`,
        provider,
        adapter: anthropic ? "anthropic" : "openai",
      });
      await seed.createModel({
        display_name: name,
        provider,
        model_name: anthropic ? "claude-3-5-haiku-20241022" : "gpt-4o-mini",
        provider_key_id: pk.id,
      });
    };
    await model(CHAT, "openai", { nonStreamBody: chatBody("priority") });
    await model(CHAT_STREAM, "openai", { streamEvents: chatStream("priority") });
    await model(CHAT_NONE, "openai", { nonStreamBody: chatBody() });
    await model(RESP, "openai", { nonStreamBody: responsesObject("flex", "completed", true) });
    await model(RESP_STREAM, "openai", { streamEvents: RESPONSES_STREAM });
    await model(MSG_FAST, "anthropic", {
      nonStreamBody: anthropicBody({ service_tier: "standard", speed: "fast" }),
    });
    await model(MSG_TIER, "anthropic", { nonStreamBody: anthropicBody({ service_tier: "priority" }) });
    await model(MSG_STREAM, "anthropic", { rawStreamFrames: ANTHROPIC_STREAM });
    // `deepseek` serves no native Responses API, so `/v1/responses` is
    // bridged onto its chat completions.
    await model(BRIDGE, "deepseek", { nonStreamBody: chatBody("priority") });
    await model(BRIDGE_NONE, "deepseek", { nonStreamBody: chatBody() });
    await model(BRIDGE_STREAM, "deepseek", { streamEvents: chatStream("scale") });

    // Seeded last: its authenticating implies every row above is loaded.
    await seed.createApiKey({ key_hash: CALLER_HASH, allowed_models: ["*"] });
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${CALLER}` },
      });
      const body = (await res.json()) as { data?: Array<{ id?: string }> };
      return res.status === 200 && (body.data ?? []).some((m) => m.id === BRIDGE_STREAM);
    });
  }, 60_000);

  afterAll(async () => {
    await app?.exit();
    await sls?.close();
    await otlp?.close();
    await Promise.all(upstreams.map((u) => u.close()));
  });

  async function call(path: string, body: Record<string, unknown>): Promise<{ text: string; requestId: string }> {
    const res = await fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER}`,
        "content-type": "application/json",
        "anthropic-version": "2023-06-01",
      },
      body: JSON.stringify(body),
    });
    const text = await res.text();
    expect(res.status, text).toBe(200);
    return { text, requestId: res.headers.get("x-aisix-request-id") ?? "" };
  }

  /** The usage row of the one request made to `model`. */
  function usageRow(model: string): Promise<Map<string, string>> {
    return waitForSlsLog(sls!, LOGSTORE, (l) => l.get("requested_model") === model, `usage row for ${model}`, 15_000);
  }

  async function attemptSpanTier(requestId: string): Promise<unknown> {
    const deadline = Date.now() + 15_000;
    while (Date.now() < deadline) {
      const span = otlp!.spans.find(
        (s) => s.attributes["aisix.request_id"] === requestId && s.attributes["gen_ai.operation.name"] !== undefined,
      );
      if (span) return span.attributes["aisix.service_tier"];
      await new Promise((r) => setTimeout(r, 50));
    }
    throw new Error(`no attempt span for ${requestId}`);
  }

  const chatRequest = (model: string, extra: Record<string, unknown> = {}) => ({
    model,
    messages: [{ role: "user", content: "hi" }],
    ...extra,
  });

  test("chat completions: the upstream's tier reaches the event, the span and the client", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    // The request asks for another tier; only the served one is recorded.
    const { text, requestId } = await call("/v1/chat/completions", chatRequest(CHAT, { service_tier: "flex" }));
    expect(JSON.parse(text).service_tier).toBe("priority");
    expect((await usageRow(CHAT)).get("service_tier")).toBe("priority");
    expect(await attemptSpanTier(requestId)).toBe("priority");
  });

  test("streamed chat completions: every chunk and the event carry the upstream's tier", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const { text, requestId } = await call(
      "/v1/chat/completions",
      chatRequest(CHAT_STREAM, { stream: true, stream_options: { include_usage: true } }),
    );
    const frames = sseFrames(text);
    expect(frames.length).toBeGreaterThan(0);
    for (const frame of frames) expect(frame.service_tier, JSON.stringify(frame)).toBe("priority");
    expect((await usageRow(CHAT_STREAM)).get("service_tier")).toBe("priority");
    expect(await attemptSpanTier(requestId)).toBe("priority");
  });

  test("an upstream that reports no tier leaves it absent everywhere", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const { text, requestId } = await call(
      "/v1/chat/completions",
      chatRequest(CHAT_NONE, { service_tier: "priority" }),
    );
    expect(JSON.parse(text)).not.toHaveProperty("service_tier");
    const row = await usageRow(CHAT_NONE);
    expect(row.get("prompt_tokens")).toBe("3");
    expect(row.has("service_tier")).toBe(false);
    expect(await attemptSpanTier(requestId)).toBeUndefined();
  });

  test("native responses: the Response object's tier is recorded", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    await call("/v1/responses", { model: RESP, input: "hi", service_tier: "auto" });
    expect((await usageRow(RESP)).get("service_tier")).toBe("flex");
  });

  test("streamed native responses: the last snapshot's tier is recorded", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    await call("/v1/responses", { model: RESP_STREAM, input: "hi", stream: true });
    expect((await usageRow(RESP_STREAM)).get("service_tier")).toBe("flex");
  });

  test("messages: a fast-mode speed wins over the standard service tier", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    await call("/v1/messages", { model: MSG_FAST, max_tokens: 16, messages: [{ role: "user", content: "hi" }] });
    expect((await usageRow(MSG_FAST)).get("service_tier")).toBe("fast");
  });

  test("messages: with no speed, usage.service_tier is recorded", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    await call("/v1/messages", { model: MSG_TIER, max_tokens: 16, messages: [{ role: "user", content: "hi" }] });
    expect((await usageRow(MSG_TIER)).get("service_tier")).toBe("priority");
  });

  test("streamed messages: the tier on message_start is recorded", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    await call("/v1/messages", {
      model: MSG_STREAM,
      max_tokens: 16,
      stream: true,
      messages: [{ role: "user", content: "hi" }],
    });
    expect((await usageRow(MSG_STREAM)).get("service_tier")).toBe("fast");
  });

  test("responses bridged to a chat upstream report the upstream's tier, not the requested one", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const { text } = await call("/v1/responses", { model: BRIDGE, input: "hi", service_tier: "flex" });
    expect(JSON.parse(text).service_tier).toBe("priority");
    expect((await usageRow(BRIDGE)).get("service_tier")).toBe("priority");
  });

  test("responses bridged to a chat upstream that reports no tier say \"default\"", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const { text } = await call("/v1/responses", { model: BRIDGE_NONE, input: "hi", service_tier: "flex" });
    expect(JSON.parse(text).service_tier).toBe("default");
    expect((await usageRow(BRIDGE_NONE)).has("service_tier")).toBe(false);
  });

  test("streamed bridged responses carry the upstream's tier on every Response object", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const { text } = await call("/v1/responses", {
      model: BRIDGE_STREAM,
      input: "hi",
      stream: true,
      service_tier: "priority",
    });
    const responses = sseFrames(text)
      .map((f) => f.response as Record<string, unknown> | undefined)
      .filter((r): r is Record<string, unknown> => r !== undefined);
    expect(responses.length).toBeGreaterThan(0);
    for (const r of responses) expect(r.service_tier).toBe("scale");
    expect((await usageRow(BRIDGE_STREAM)).get("service_tier")).toBe("scale");
  });
});
