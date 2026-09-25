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

// E2E: a monitor-mode output guardrail on a streamed response judges ALL of
// the generated text, on every streamed route. A monitor-only chain never
// holds the stream back, so no hold cap applies to what it scans: a phrase
// that arrives after 256 KiB of output is still a would-be block on the
// request's usage event.
//
// The row is `kind: custom`, which judges the flattened text, so the chat
// route's tool-call arguments reach it through the same text the other
// routes scan.

const CALLER = "sk-monitor-full-output-scan-e2e";
const CREDENTIAL_REF = "mock";
const LOGSTORE = "monitor-full-output-scan-events";
const GUARD_NAME = "monitor-full-output-scan";
const hash = (s: string) => createHash("sha256").update(s).digest("hex");

const MARKER = "latemonitormarker";
const SCRIPT = `
export function checkOutput(ctx) {
  return ctx.text.includes("${MARKER}") ? { action: "block" } : { action: "none" };
}`;

// 220 × 1250 = 275 000 bytes of clean output ahead of the marker: past the
// 256 KiB (262 144-byte) default hold cap.
const PIECE = "z".repeat(1250);
const pieces = [...Array.from({ length: 220 }, () => PIECE), ` then ${MARKER}`];

const chatChunk = (delta: Record<string, unknown>, finish: string | null = null) =>
  JSON.stringify({
    id: "chatcmpl-late",
    object: "chat.completion.chunk",
    created: 1,
    model: "gpt-4o-mini",
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
const chatDone = [
  JSON.stringify({
    id: "chatcmpl-late",
    object: "chat.completion.chunk",
    created: 1,
    model: "gpt-4o-mini",
    choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
    usage: { prompt_tokens: 5, completion_tokens: 12, total_tokens: 17 },
  }),
  "[DONE]",
];

const chatContent = [
  chatChunk({ role: "assistant" }),
  ...pieces.map((content) => chatChunk({ content })),
  ...chatDone,
];

const chatToolCall = [
  chatChunk({
    role: "assistant",
    tool_calls: [{ index: 0, id: "call_late", type: "function", function: { name: "lookup", arguments: "" } }],
  }),
  ...pieces.map((args) => chatChunk({ tool_calls: [{ index: 0, function: { arguments: args } }] })),
  ...chatDone,
];

const responsesText = [
  JSON.stringify({ type: "response.created", response: { id: "resp_late" } }),
  ...pieces.map((delta) => JSON.stringify({ type: "response.output_text.delta", delta })),
  JSON.stringify({
    type: "response.completed",
    response: { id: "resp_late", status: "completed", usage: { input_tokens: 5, output_tokens: 12 } },
  }),
  "[DONE]",
];

const anthropicText = [
  JSON.stringify({
    type: "message_start",
    message: {
      id: "msg_late",
      role: "assistant",
      content: [],
      model: "claude-3-5-haiku-20241022",
      stop_reason: null,
      usage: { input_tokens: 5, output_tokens: 1 },
    },
  }),
  JSON.stringify({ type: "content_block_start", index: 0, content_block: { type: "text", text: "" } }),
  ...pieces.map((text) => JSON.stringify({ type: "content_block_delta", index: 0, delta: { type: "text_delta", text } })),
  JSON.stringify({ type: "content_block_stop", index: 0 }),
  JSON.stringify({ type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 12 } }),
  JSON.stringify({ type: "message_stop" }),
];

// Streamed transcription deltas; the terminal event carries usage only, so
// the transcript is the assembled deltas.
const transcriptFrames = [
  ...pieces.map((delta) => `data: ${JSON.stringify({ type: "transcript.text.delta", delta })}\n\n`),
  `data: ${JSON.stringify({
    type: "transcript.text.done",
    usage: { type: "tokens", input_tokens: 26, output_tokens: 12, total_tokens: 38 },
  })}\n\n`,
  "data: [DONE]\n\n",
];

interface Route {
  name: string;
  provider: "anthropic" | "openai";
  upstream: { streamEvents?: string[]; rawStreamFrames?: string[] };
  // `apis: {}` declares an OpenAI-compatible endpoint with no
  // `/v1/responses`, which puts `/v1/responses` on the Chat bridge.
  apis?: Record<string, never>;
  send: (proxyUrl: string, model: string) => Promise<Response>;
}

const json = (proxyUrl: string, path: string, body: Record<string, unknown>) =>
  fetch(`${proxyUrl}${path}`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: `Bearer ${CALLER}`,
      "x-api-key": CALLER,
      "anthropic-version": "2023-06-01",
    },
    body: JSON.stringify({ ...body, stream: true }),
  });

const ROUTES: Route[] = [
  {
    name: "responses-native",
    provider: "openai",
    upstream: { streamEvents: responsesText },
    send: (u, model) => json(u, "/v1/responses", { model, input: "go" }),
  },
  {
    name: "responses-bridge",
    provider: "openai",
    upstream: { streamEvents: chatContent },
    apis: {},
    send: (u, model) => json(u, "/v1/responses", { model, input: "go" }),
  },
  {
    name: "chat-tool-call",
    provider: "openai",
    upstream: { streamEvents: chatToolCall },
    send: (u, model) => json(u, "/v1/chat/completions", { model, messages: [{ role: "user", content: "go" }] }),
  },
  {
    name: "chat-content",
    provider: "openai",
    upstream: { streamEvents: chatContent },
    send: (u, model) => json(u, "/v1/chat/completions", { model, messages: [{ role: "user", content: "go" }] }),
  },
  {
    name: "messages-native",
    provider: "anthropic",
    upstream: { streamEvents: anthropicText },
    send: (u, model) => json(u, "/v1/messages", { model, max_tokens: 64, messages: [{ role: "user", content: "go" }] }),
  },
  {
    name: "transcription",
    provider: "openai",
    upstream: { rawStreamFrames: transcriptFrames },
    send: (u, model) => {
      const form = new FormData();
      form.set("model", model);
      form.set("stream", "true");
      form.set("file", new Blob([new Uint8Array([0x49, 0x44, 0x33])], { type: "audio/mpeg" }), "a.mp3");
      return fetch(`${u}/v1/audio/transcriptions`, {
        method: "POST",
        headers: { authorization: `Bearer ${CALLER}` },
        body: form,
      });
    },
  },
];

describe("monitor-mode output guardrail scans all of a streamed response", () => {
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
      name: "sls-monitor-full-output-scan",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: "aisix-e2e-obs",
      logstore: LOGSTORE,
      credential_ref: CREDENTIAL_REF,
      content_mode: "metadata_only",
    });
    const monitor = await seed.createGuardrail(
      {
        name: GUARD_NAME,
        enabled: true,
        kind: "custom",
        hook_point: "output",
        enforcement_mode: "monitor",
        output_fail_open: false,
        timeout_ms: 5000,
        script: SCRIPT,
      },
      { attach: false },
    );

    const models: string[] = [];
    for (const route of ROUTES) {
      const upstream = await startOpenAiUpstream(route.upstream);
      upstreams.push(upstream);
      const display = `late-${route.name}`;
      const pk = await seed.createProviderKey({
        display_name: `${display}-pk`,
        secret: "sk-mock",
        api_base: route.provider === "anthropic" ? upstream.baseUrl : `${upstream.baseUrl}/v1`,
        ...(route.apis ? { apis: route.apis } : {}),
      });
      const model = await seed.createModel({
        display_name: display,
        provider: route.provider,
        model_name:
          route.provider === "anthropic"
            ? "claude-3-5-haiku-20241022"
            : route.name === "transcription"
              ? "gpt-4o-transcribe"
              : "gpt-4o-mini",
        provider_key_id: pk.id,
      });
      await seed.attachGuardrailToModel(monitor.id as string, model.id as string);
      models.push(display);
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

  for (const route of ROUTES) {
    test(`${route.name}: a trigger past 256 KiB of output is a monitor hit`, async (ctx) => {
      if (!etcdReachable || !app || !sls) {
        ctx.skip();
        return;
      }
      const model = `late-${route.name}`;
      const res = await route.send(app.proxyUrl, model);
      expect(res.status).toBe(200);
      const body = await res.text();
      expect(body, "monitor mode releases the whole stream").toContain(MARKER);
      expect(body).not.toContain("blocked by content policy");
      expect(body).not.toContain("output_buffer_exceeded");
      const event = await waitForSlsLog(
        sls,
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
        expect.objectContaining({ action: "would_block", hook: "output", guardrail_name: GUARD_NAME }),
      );
    });
  }
});
