import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
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

// E2E: a `kind: custom` OUTPUT guardrail on a STREAMED response, on every
// streamed LLM route. The script is the operator's policy over the generated
// content, so on each route it must:
//   - see that content (a phrase the script blocks on is actually blocked);
//   - decide before any of it reaches the caller (the refused content, and
//     the clean text streamed ahead of it, never appear on the wire);
//   - when it faults on a fail-closed row, refuse before any content goes
//     out rather than after.
// A clean stream still reaches the caller whole, and a monitor-mode row —
// which never holds a stream back — still judges what was streamed, so its
// would-be block reaches the usage event.
//
// The row uses the kind's default streaming mode, so nothing here depends on
// the operator having picked `buffer_full`.

const CALLER = "sk-custom-streamed-output-e2e";
const CREDENTIAL_REF = "mock";
const LOGSTORE = "custom-streamed-output-events";
const hash = (s: string) => createHash("sha256").update(s).digest("hex");

const LEAD = "Sure, here is the plan you asked for. ";
const MARKER = "cstmstreamedmarker";
const CLEAN_TAIL = "Nothing else to add.";

const SCRIPT_BLOCK = `
export function checkOutput(ctx) {
  return ctx.text.includes("${MARKER}") ? { action: "block" } : { action: "none" };
}`;
// An action the gateway does not know: a script fault.
const SCRIPT_FAULT = `export function checkOutput() { return { action: "permit" }; }`;

// The same generated text in each wire protocol: the lead, then the marker
// split across two deltas (or, for the clean stream, a clean tail).
const pieces = (risky: boolean) => (risky ? [LEAD, "cstmstreamed", "marker"] : [LEAD, CLEAN_TAIL]);

const anthropicEvents = (risky: boolean) => [
  JSON.stringify({
    type: "message_start",
    message: {
      id: "msg_custom_stream",
      role: "assistant",
      content: [],
      model: "claude-3-5-haiku-20241022",
      stop_reason: null,
      usage: { input_tokens: 5, output_tokens: 1 },
    },
  }),
  JSON.stringify({
    type: "content_block_start",
    index: 0,
    content_block: { type: "text", text: "" },
  }),
  ...pieces(risky).map((text) =>
    JSON.stringify({
      type: "content_block_delta",
      index: 0,
      delta: { type: "text_delta", text },
    }),
  ),
  JSON.stringify({ type: "content_block_stop", index: 0 }),
  JSON.stringify({
    type: "message_delta",
    delta: { stop_reason: "end_turn" },
    usage: { output_tokens: 12 },
  }),
  JSON.stringify({ type: "message_stop" }),
];

const chatEvents = (risky: boolean) => [
  JSON.stringify({
    id: "chatcmpl-custom-stream",
    object: "chat.completion.chunk",
    created: 1,
    model: "gpt-4o-mini",
    choices: [{ index: 0, delta: { role: "assistant" }, finish_reason: null }],
  }),
  ...pieces(risky).map((content) =>
    JSON.stringify({
      id: "chatcmpl-custom-stream",
      object: "chat.completion.chunk",
      created: 1,
      model: "gpt-4o-mini",
      choices: [{ index: 0, delta: { content }, finish_reason: null }],
    }),
  ),
  JSON.stringify({
    id: "chatcmpl-custom-stream",
    object: "chat.completion.chunk",
    created: 1,
    model: "gpt-4o-mini",
    choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
    usage: { prompt_tokens: 5, completion_tokens: 12, total_tokens: 17 },
  }),
  "[DONE]",
];

const responsesEvents = (risky: boolean) => [
  JSON.stringify({ type: "response.created", response: { id: "resp_custom_stream" } }),
  ...pieces(risky).map((delta) => JSON.stringify({ type: "response.output_text.delta", delta })),
  JSON.stringify({
    type: "response.completed",
    response: {
      id: "resp_custom_stream",
      status: "completed",
      usage: { input_tokens: 5, output_tokens: 12 },
    },
  }),
  "[DONE]",
];

type Wire = "anthropic" | "chat" | "responses";

interface Route {
  name: string;
  path: string;
  wire: Wire;
  provider: "anthropic" | "openai";
  // `apis: {}` declares an OpenAI-compatible endpoint with no
  // `/v1/responses`, which puts `/v1/responses` on the Chat bridge.
  apis?: Record<string, never>;
}

const ROUTES: Route[] = [
  { name: "messages-native", path: "/v1/messages", wire: "anthropic", provider: "anthropic" },
  // An OpenAI-shape upstream answering `/v1/messages`: the Anthropic stream
  // is encoded by the gateway from Chat chunks.
  { name: "messages-bridge", path: "/v1/messages", wire: "chat", provider: "openai" },
  { name: "chat", path: "/v1/chat/completions", wire: "chat", provider: "openai" },
  { name: "responses-native", path: "/v1/responses", wire: "responses", provider: "openai" },
  { name: "responses-bridge", path: "/v1/responses", wire: "chat", provider: "openai", apis: {} },
];

const EVENTS: Record<Wire, (risky: boolean) => string[]> = {
  anthropic: anthropicEvents,
  chat: chatEvents,
  responses: responsesEvents,
};

describe("custom output guardrail on streamed responses", () => {
  let app: SpawnedApp | undefined;
  let sls: MockSls | undefined;
  const upstreams: OpenAiUpstream[] = [];
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    sls = await startMockSls();
    app = await spawnApp({
      extraEnv: {
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: "mock-akid",
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: "mock-secret",
      },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createObservabilityExporter({
      name: "sls-custom-streamed-output",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: "aisix-e2e-obs",
      logstore: LOGSTORE,
      credential_ref: CREDENTIAL_REF,
      content_mode: "metadata_only",
    });

    const block = await seed.createGuardrail(
      {
        name: "custom-stream-block",
        enabled: true,
        kind: "custom",
        hook_point: "output",
        output_fail_open: false,
        timeout_ms: 5000,
        script: SCRIPT_BLOCK,
      },
      { attach: false },
    );
    const fault = await seed.createGuardrail(
      {
        name: "custom-stream-fault",
        enabled: true,
        kind: "custom",
        hook_point: "output",
        output_fail_open: false,
        timeout_ms: 5000,
        script: SCRIPT_FAULT,
      },
      { attach: false },
    );
    const monitor = await seed.createGuardrail(
      {
        name: "custom-stream-monitor",
        enabled: true,
        kind: "custom",
        hook_point: "output",
        enforcement_mode: "monitor",
        output_fail_open: false,
        timeout_ms: 5000,
        script: SCRIPT_BLOCK,
      },
      { attach: false },
    );

    const models: string[] = [];
    for (const route of ROUTES) {
      // `risky` streams the marker; `clean` does not.
      for (const [variant, risky, guardrail] of [
        ["block", true, block],
        ["clean", false, block],
        ["fault", true, fault],
        ["monitor", true, monitor],
      ] as const) {
        // Trickled, so a relay that forwards frames as they arrive would
        // put the lead on the wire well before the stream ends.
        const upstream = await startOpenAiUpstream({
          streamEvents: EVENTS[route.wire](risky),
          eventDelayMs: 20,
        });
        upstreams.push(upstream);
        const display = `${route.name}-${variant}`;
        const pk = await seed.createProviderKey({
          display_name: `${display}-pk`,
          secret: "sk-mock",
          api_base: route.provider === "anthropic" ? upstream.baseUrl : `${upstream.baseUrl}/v1`,
          ...(route.apis ? { apis: route.apis } : {}),
        });
        const model = await seed.createModel({
          display_name: display,
          provider: route.provider,
          model_name: route.provider === "anthropic" ? "claude-3-5-haiku-20241022" : "gpt-4o-mini",
          provider_key_id: pk.id,
        });
        await seed.attachGuardrailToModel(guardrail.id as string, model.id as string);
        models.push(display);
      }
    }

    // Seeded last: its key authenticating implies the whole seed is live.
    await seed.createApiKey({ key_hash: hash(CALLER), allowed_models: models });
    await waitConfigPropagation(
      async () => (await new ProxyClient(app!.proxyUrl, CALLER).listModels()).status === 200,
    );
  }, 120_000);

  afterAll(async () => {
    await app?.exit();
    await sls?.close();
    await Promise.all(upstreams.map((u) => u.close()));
  });

  const stream = async (route: Route, model: string) => {
    const body =
      route.wire === "anthropic" || route.path === "/v1/messages"
        ? { model, max_tokens: 64, stream: true, messages: [{ role: "user", content: "plan it" }] }
        : route.path === "/v1/chat/completions"
          ? { model, stream: true, messages: [{ role: "user", content: "plan it" }] }
          : { model, stream: true, input: "plan it" };
    const res = await fetch(`${app!.proxyUrl}${route.path}`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${CALLER}`,
        "x-api-key": CALLER,
        "anthropic-version": "2023-06-01",
      },
      body: JSON.stringify(body),
    });
    return { status: res.status, body: await res.text() };
  };

  // A refusal is either an HTTP 422 (nothing was streamed yet, so the route
  // can still answer with a status) or a 200 stream carrying an SSE error.
  const expectRefusedBeforeContent = (
    res: { status: number; body: string },
    message: string,
  ) => {
    expect([200, 422]).toContain(res.status);
    if (res.status === 200) expect(res.body).toContain("error");
    expect(res.body).toContain(message);
    expect(res.body, "the refused content stays off the wire").not.toContain(MARKER);
    expect(res.body, "nothing is sent ahead of the verdict").not.toContain(LEAD.trim());
  };

  for (const route of ROUTES) {
    test(`${route.name}: the script sees the streamed content and blocks it before any is sent`, async (ctx) => {
      if (!etcdReachable) ctx.skip();
      expectRefusedBeforeContent(
        await stream(route, `${route.name}-block`),
        "blocked by content policy",
      );
    });

    test(`${route.name}: a fail-closed script fault refuses before any content is sent`, async (ctx) => {
      if (!etcdReachable) ctx.skip();
      expectRefusedBeforeContent(await stream(route, `${route.name}-fault`), "could not evaluate");
    });

    test(`${route.name}: a clean stream is delivered whole`, async (ctx) => {
      if (!etcdReachable) ctx.skip();
      const res = await stream(route, `${route.name}-clean`);
      expect(res.status).toBe(200);
      expect(res.body).toContain(LEAD.trim());
      expect(res.body).toContain(CLEAN_TAIL);
      expect(res.body).not.toContain("could not evaluate");
      expect(res.body).not.toContain("blocked by content policy");
    });

    test(`${route.name}: a monitor-mode script judges what was streamed`, async (ctx) => {
      if (!etcdReachable || !sls) ctx.skip();
      const model = `${route.name}-monitor`;
      const res = await stream(route, model);
      expect(res.status).toBe(200);
      expect(res.body).toContain(LEAD.trim());
      expect(res.body).not.toContain("blocked by content policy");
      const event = await waitForSlsLog(
        sls!,
        LOGSTORE,
        (log) => log.get("requested_model") === model,
        `usage event for ${model}`,
      );
      const hits = JSON.parse(event.get("guardrail_monitor_hits") ?? "[]") as Array<{
        action: string;
        hook: string;
        guardrail_name: string;
      }>;
      expect(hits).toContainEqual(
        expect.objectContaining({
          action: "would_block",
          hook: "output",
          guardrail_name: "custom-stream-monitor",
        }),
      );
    });
  }
});
