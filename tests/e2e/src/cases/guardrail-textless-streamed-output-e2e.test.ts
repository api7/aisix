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

// E2E: the output guardrail chain decides a response that generated no text,
// on every streamed route and on the buffered transcript. "Nothing to scan" is
// not "nothing to decide": a `kind: custom` script can block unconditionally,
// while a kind that matches on text has matched nothing and lets it through.

const CALLER = "sk-textless-streamed-output-e2e";
const CREDENTIAL_REF = "mock";
const LOGSTORE = "textless-streamed-output-events";
const hash = (s: string) => createHash("sha256").update(s).digest("hex");
const REFUSAL = "blocked by content policy";
const SCRIPT = `export function checkOutput() { return { action: "block" }; }`;

const anthropicFrames = [
  JSON.stringify({
    type: "message_start",
    message: {
      id: "msg_textless",
      role: "assistant",
      content: [],
      model: "claude-3-5-haiku-20241022",
      stop_reason: null,
      usage: { input_tokens: 5, output_tokens: 1 },
    },
  }),
  JSON.stringify({ type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 1 } }),
  JSON.stringify({ type: "message_stop" }),
];
const chatChunk = (delta: Record<string, unknown>, finish_reason: string | null) =>
  JSON.stringify({
    id: "chatcmpl-textless",
    object: "chat.completion.chunk",
    created: 1,
    model: "gpt-4o-mini",
    choices: [{ index: 0, delta, finish_reason }],
    ...(finish_reason ? { usage: { prompt_tokens: 5, completion_tokens: 1, total_tokens: 6 } } : {}),
  });
const chatFrames = [chatChunk({ role: "assistant" }, null), chatChunk({}, "stop"), "[DONE]"];
const responsesFrames = [
  JSON.stringify({ type: "response.created", response: { id: "resp_textless" } }),
  JSON.stringify({
    type: "response.completed",
    response: { id: "resp_textless", status: "completed", usage: { input_tokens: 5, output_tokens: 1 } },
  }),
  "[DONE]",
];
const transcriptFrames = [
  `data: ${JSON.stringify({
    type: "transcript.text.done",
    usage: { type: "tokens", input_tokens: 26, output_tokens: 1, total_tokens: 27 },
  })}\n\n`,
  "data: [DONE]\n\n",
];

const headers = {
  "content-type": "application/json",
  authorization: `Bearer ${CALLER}`,
  "x-api-key": CALLER,
  "anthropic-version": "2023-06-01",
};
const json = (proxyUrl: string, path: string, body: Record<string, unknown>) =>
  fetch(`${proxyUrl}${path}`, { method: "POST", headers, body: JSON.stringify(body) });
const transcribe = (proxyUrl: string, model: string, stream: boolean) => {
  const form = new FormData();
  form.set("model", model);
  if (stream) form.set("stream", "true");
  form.set("file", new Blob([new Uint8Array([0x49, 0x44, 0x33])], { type: "audio/mpeg" }), "a.mp3");
  return fetch(`${proxyUrl}/v1/audio/transcriptions`, {
    method: "POST",
    headers: { authorization: `Bearer ${CALLER}` },
    body: form,
  });
};
const HI = [{ role: "user", content: "hi" }];

interface Route {
  name: string;
  provider: "anthropic" | "openai";
  upstream: { streamEvents?: string[]; rawStreamFrames?: string[]; nonStreamBody?: unknown };
  // `apis: {}` declares an OpenAI-compatible endpoint with no
  // `/v1/responses`, which puts `/v1/responses` on the Chat bridge.
  apis?: Record<string, never>;
  send: (proxyUrl: string, model: string) => Promise<Response>;
}

const ROUTES: Route[] = [
  {
    name: "chat",
    provider: "openai",
    upstream: { streamEvents: chatFrames },
    send: (u, model) => json(u, "/v1/chat/completions", { model, stream: true, messages: HI }),
  },
  {
    name: "messages-native",
    provider: "anthropic",
    upstream: { streamEvents: anthropicFrames },
    send: (u, model) => json(u, "/v1/messages", { model, stream: true, max_tokens: 64, messages: HI }),
  },
  {
    name: "messages-bridge",
    provider: "openai",
    upstream: { streamEvents: chatFrames },
    send: (u, model) => json(u, "/v1/messages", { model, stream: true, max_tokens: 64, messages: HI }),
  },
  {
    name: "responses-native",
    provider: "openai",
    upstream: { streamEvents: responsesFrames },
    send: (u, model) => json(u, "/v1/responses", { model, stream: true, input: "hi" }),
  },
  {
    name: "responses-bridge",
    provider: "openai",
    upstream: { streamEvents: chatFrames },
    apis: {},
    send: (u, model) => json(u, "/v1/responses", { model, stream: true, input: "hi" }),
  },
  {
    name: "transcription-stream",
    provider: "openai",
    upstream: { rawStreamFrames: transcriptFrames },
    send: (u, model) => transcribe(u, model, true),
  },
  {
    name: "transcription",
    provider: "openai",
    upstream: { nonStreamBody: { text: "" } },
    send: (u, model) => transcribe(u, model, false),
  },
];
// An enforcing row sends a transcription stream down the buffered path, which
// `transcription` covers; only a monitor row reaches its end-of-stream scan.
const ENFORCED = ROUTES.filter((route) => route.name !== "transcription-stream");

describe("output guardrails decide a response with no text", () => {
  let app: SpawnedApp | undefined;
  let sls: MockSls | undefined;
  const upstreams: OpenAiUpstream[] = [];

  beforeAll(async () => {
    const etcd = new EtcdClient();
    if (!(await etcd.ping())) return;
    sls = await startMockSls();
    app = await spawnApp({
      extraEnv: {
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: "mock-akid",
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: "mock-secret",
      },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createObservabilityExporter({
      name: "sls-textless",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: "aisix-e2e-obs",
      logstore: LOGSTORE,
      credential_ref: CREDENTIAL_REF,
      content_mode: "metadata_only",
    });
    const custom = { kind: "custom", output_fail_open: false, timeout_ms: 5000, script: SCRIPT };
    const rows = {
      block: await seed.createGuardrail(
        { name: "textless-block", enabled: true, hook_point: "output", ...custom },
        { attach: false },
      ),
      monitor: await seed.createGuardrail(
        { name: "textless-monitor", enabled: true, hook_point: "output", enforcement_mode: "monitor", ...custom },
        { attach: false },
      ),
      keyword: await seed.createGuardrail(
        {
          name: "textless-keyword",
          enabled: true,
          hook_point: "output",
          kind: "keyword",
          patterns: [{ kind: "literal", value: "never-in-this-stream" }],
        },
        { attach: false },
      ),
    };

    const models: string[] = [];
    for (const route of ROUTES) {
      for (const [variant, guardrail] of Object.entries(rows)) {
        const upstream = await startOpenAiUpstream(route.upstream);
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

  for (const route of ENFORCED) {
    test(`${route.name}: a script that blocks unconditionally refuses it`, async (ctx) => {
      if (!app) return ctx.skip();
      const res = await route.send(app.proxyUrl, `${route.name}-block`);
      expect([200, 422]).toContain(res.status);
      expect(await res.text()).toContain(REFUSAL);
    });
  }

  for (const route of ROUTES) {
    test(`${route.name}: a text-matching guardrail lets it through`, async (ctx) => {
      if (!app) return ctx.skip();
      const res = await route.send(app.proxyUrl, `${route.name}-keyword`);
      expect(res.status).toBe(200);
      expect(await res.text()).not.toContain(REFUSAL);
    });

    test(`${route.name}: a monitor-mode script records its verdict`, async (ctx) => {
      if (!app || !sls) return ctx.skip();
      const model = `${route.name}-monitor`;
      const res = await route.send(app.proxyUrl, model);
      expect(res.status).toBe(200);
      expect(await res.text()).not.toContain(REFUSAL);
      const event = await waitForSlsLog(
        sls,
        LOGSTORE,
        (log) => log.get("requested_model") === model,
        `usage event for ${model}`,
      );
      const hits = JSON.parse(event.get("guardrail_monitor_hits") ?? "[]") as unknown[];
      expect(hits).toContainEqual(
        expect.objectContaining({ action: "would_block", hook: "output", guardrail_name: "textless-monitor" }),
      );
    });
  }
});
