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
import type { OpenAiUpstreamOptions } from "../harness/upstream-openai.js";

// What the usage record says about a stream that did not end cleanly after
// its `200` headers went out.
//
// An upstream failure after the headers — a dropped connection, an in-band
// error event, a stream that carried nothing — is recorded with the status
// the same failure gets before the headers, and the failure itself as the
// row's error class and message. It used to be recorded as a `200` with no
// error, which the console counts as a success. A caller that walks away
// mid-stream is a `499` on every family, the ensemble judge's stream
// included.
//
// Observed through the SLS exporter, which receives the usage row as the
// control plane does.

const CALLER_PLAINTEXT = "sk-stream-failure-usage-status";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");
const CREDENTIAL_REF = "mock";
const SLS_PROJECT = "stream-failure-proj";
const LOGSTORE = "stream-failure-store";

const USAGE = { prompt_tokens: 5, completion_tokens: 3, total_tokens: 8 };

function chunk(delta: Record<string, unknown>, finish: string | null = null): string {
  return JSON.stringify({
    id: "chatcmpl-sf",
    object: "chat.completion.chunk",
    created: 1_700_000_000,
    model: "relay-mini",
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
}

function sseFrame(event: string, data: unknown): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

// A stream that sends one piece of content, then loses its connection.
const CUT_AFTER_CONTENT: OpenAiUpstreamOptions = {
  streamEvents: [chunk({ content: "half an ans" }), chunk({ content: "wer" }), chunk({}, "stop"), "[DONE]"],
  disconnectAfterEvents: 1,
  eventDelayMs: 50,
};

// A stream slow enough for the caller to read its first event and leave.
const TRICKLE: OpenAiUpstreamOptions = {
  eventDelayMs: 500,
  streamEvents: [
    chunk({ role: "assistant", content: "one" }),
    chunk({ content: "two" }),
    chunk({ content: "three" }),
    chunk({}, "stop"),
    "[DONE]",
  ],
};

const ANTHROPIC_IN_BAND_ERROR_FRAMES = [
  sseFrame("message_start", {
    type: "message_start",
    message: {
      id: "msg_sf",
      type: "message",
      role: "assistant",
      content: [],
      model: "claude-3-5-haiku-20241022",
      stop_reason: null,
      usage: { input_tokens: 5, output_tokens: 1 },
    },
  }),
  sseFrame("content_block_start", {
    type: "content_block_start",
    index: 0,
    content_block: { type: "text", text: "" },
  }),
  sseFrame("content_block_delta", {
    type: "content_block_delta",
    index: 0,
    delta: { type: "text_delta", text: "partial" },
  }),
  sseFrame("error", {
    type: "error",
    error: { type: "rate_limit_error", message: "Number of request tokens has exceeded your rate limit." },
  }),
];

const RESPONSES_FAILED_FRAMES = [
  sseFrame("response.created", {
    type: "response.created",
    sequence_number: 0,
    response: { id: "resp_sf", object: "response", status: "in_progress", model: "gpt-4o", output: [] },
  }),
  sseFrame("response.output_text.delta", {
    type: "response.output_text.delta",
    sequence_number: 1,
    item_id: "msg_sf",
    output_index: 0,
    content_index: 0,
    delta: "partial",
  }),
  sseFrame("response.failed", {
    type: "response.failed",
    sequence_number: 2,
    response: {
      id: "resp_sf",
      object: "response",
      status: "failed",
      model: "gpt-4o",
      output: [],
      error: { code: "server_error", message: "The model failed partway through the response." },
      usage: null,
    },
  }),
];

// A streamed transcription that delivers one delta, then loses its
// connection.
const TRANSCRIPT_CUT: OpenAiUpstreamOptions = {
  rawStreamFrames: [
    `data: ${JSON.stringify({ type: "transcript.text.delta", delta: "hello" })}\n\n`,
    `data: ${JSON.stringify({ type: "transcript.text.delta", delta: " world" })}\n\n`,
    "data: [DONE]\n\n",
  ],
  disconnectAfterEvents: 1,
  eventDelayMs: 50,
};

const ROUTE = "sf-route";
const ROUTE_PREFIX = "/passthrough/sf";

interface Seeded {
  upstream: OpenAiUpstreamOptions;
  /** Attach a blocking output guardrail, so the response is held back. */
  holdBack?: boolean;
  /** `chat` models are OpenAI-wire; `responses-bridge` ones declare no
   *  `/v1/responses`, so that surface is translated through chat. */
  kind: "chat" | "responses-bridge" | "responses-native" | "anthropic";
}

const MODELS: Record<string, Seeded> = {
  "sf-chat-cut": { upstream: CUT_AFTER_CONTENT, kind: "chat" },
  "sf-messages-bridge-cut": { upstream: CUT_AFTER_CONTENT, kind: "chat" },
  "sf-messages-native-error": {
    upstream: { rawStreamFrames: ANTHROPIC_IN_BAND_ERROR_FRAMES },
    kind: "anthropic",
  },
  "sf-responses-bridge-cut": { upstream: CUT_AFTER_CONTENT, kind: "responses-bridge" },
  "sf-responses-bridge-empty": { upstream: { streamEvents: ["[DONE]"] }, kind: "responses-bridge" },
  "sf-responses-bridge-trickle": { upstream: TRICKLE, kind: "responses-bridge" },
  "sf-responses-native-failed": {
    upstream: { rawStreamFrames: RESPONSES_FAILED_FRAMES },
    kind: "responses-native",
  },
  "sf-responses-native-failed-held": {
    upstream: { rawStreamFrames: RESPONSES_FAILED_FRAMES },
    kind: "responses-native",
    holdBack: true,
  },
  "sf-ens-member": {
    upstream: {
      nonStreamBody: {
        id: "chatcmpl-sf-member",
        object: "chat.completion",
        created: 1_700_000_000,
        model: "relay-mini",
        choices: [{ index: 0, message: { role: "assistant", content: "a view" }, finish_reason: "stop" }],
        usage: USAGE,
      },
    },
    kind: "chat",
  },
  "sf-ens-judge": { upstream: TRICKLE, kind: "chat" },
  "sf-transcribe-cut": { upstream: TRANSCRIPT_CUT, kind: "chat" },
};
const ENSEMBLE = "sf-ensemble";

describe("usage status of a stream that fails after its 200 headers", () => {
  let etcdReachable = false;
  let app: SpawnedApp | undefined;
  let sls: MockSls | undefined;
  const upstreams: OpenAiUpstream[] = [];

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
      name: "sls-stream-failure",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: SLS_PROJECT,
      logstore: LOGSTORE,
      credential_ref: CREDENTIAL_REF,
      content_mode: "metadata_only",
    });
    for (const [model, { upstream: opts, kind, holdBack }] of Object.entries(MODELS)) {
      const upstream = await startOpenAiUpstream(opts);
      upstreams.push(upstream);
      if (kind === "anthropic") {
        // The Anthropic adapter appends `/v1/messages` to the bare host.
        const pk = await seed.createProviderKey({
          display_name: `${model}-pk`,
          secret: "sk-ant-mock",
          api_base: upstream.baseUrl,
          provider: "anthropic",
          adapter: "anthropic",
        });
        await seed.createModel({
          display_name: model,
          provider: "anthropic",
          model_name: "claude-3-5-haiku-20241022",
          provider_key_id: pk.id,
        });
        continue;
      }
      const pk = await seed.createProviderKey({
        display_name: `${model}-pk`,
        secret: "sk-mock",
        api_base: `${upstream.baseUrl}/v1`,
        ...(kind === "responses-bridge" ? { apis: {} } : {}),
      });
      const created = await seed.createModel({
        display_name: model,
        provider: "openai",
        model_name: "relay-compat-x",
        provider_key_id: pk.id,
      });
      if (holdBack) {
        const guardrail = await seed.createGuardrail(
          {
            name: `${model}-output-block`,
            enabled: true,
            hook_point: "output",
            kind: "keyword",
            patterns: [{ kind: "literal", value: "a phrase the upstream never says" }],
          },
          { attach: false },
        );
        await seed.attachGuardrailToModel(guardrail.id, created.id);
      }
    }
    // A passthrough route onto an upstream that drops its stream part-way.
    const routeUpstream = await startOpenAiUpstream(CUT_AFTER_CONTENT);
    upstreams.push(routeUpstream);
    const routePk = await seed.createProviderKey({
      display_name: "sf-route-pk",
      secret: "sk-mock",
      api_base: `${routeUpstream.baseUrl}/v1`,
    });
    await seed.createPassthroughRoute({
      name: ROUTE,
      path_prefix: ROUTE_PREFIX,
      target_url: routeUpstream.baseUrl,
      provider_key_id: routePk.id,
    });
    await seed.createModel({
      display_name: ENSEMBLE,
      ensemble: {
        panel: [{ model: "sf-ens-member" }],
        judge: { model: "sf-ens-judge" },
        min_responses: 1,
      },
    });
    // Seeded last: the key authenticating implies the whole seed set is in
    // the snapshot.
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: [...Object.keys(MODELS), ENSEMBLE],
      allowed_routes: ["*"],
    });
    const probe = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    await waitConfigPropagation(async () => (await probe.listModels()).status === 200);
  });

  afterAll(async () => {
    await app?.exit();
    for (const u of upstreams) await u.close();
    await sls?.close();
  });

  const BODIES: Record<string, (model: string) => Record<string, unknown>> = {
    "/v1/chat/completions": (model) => ({
      model,
      messages: [{ role: "user", content: "hi" }],
      stream: true,
    }),
    "/v1/messages": (model) => ({
      model,
      max_tokens: 64,
      messages: [{ role: "user", content: "hi" }],
      stream: true,
    }),
    "/v1/responses": (model) => ({ model, input: "hi", stream: true }),
  };

  async function streamToEnd(path: string, model: string): Promise<string> {
    const res = await fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: { authorization: `Bearer ${CALLER_PLAINTEXT}`, "content-type": "application/json" },
      body: JSON.stringify(BODIES[path]!(model)),
    });
    expect(res.status, "the stream's headers went out as a 200").toBe(200);
    const requestId = res.headers.get("x-aisix-request-id") ?? "";
    expect(requestId).not.toBe("");
    // A relay that ends on a transport error aborts the body; that is the
    // failure under test, not the caller's.
    await res.text().catch(() => undefined);
    return requestId;
  }

  async function abandonAfterFirstRead(path: string, model: string): Promise<string> {
    const controller = new AbortController();
    const res = await fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: { authorization: `Bearer ${CALLER_PLAINTEXT}`, "content-type": "application/json" },
      body: JSON.stringify(BODIES[path]!(model)),
      signal: controller.signal,
    });
    expect(res.status).toBe(200);
    const requestId = res.headers.get("x-aisix-request-id") ?? "";
    expect(requestId).not.toBe("");
    const reader = res.body!.getReader();
    const first = await reader.read();
    expect(first.done, "the stream delivered nothing to abandon").toBe(false);
    controller.abort();
    await reader.cancel().catch((err: unknown) => {
      if (!(err instanceof Error) || err.name !== "AbortError") throw err;
    });
    return requestId;
  }

  async function usageRow(
    requestId: string,
    pred: (log: Map<string, string>) => boolean = () => true,
  ): Promise<Map<string, string>> {
    return waitForSlsLog(
      sls!,
      LOGSTORE,
      (log) => log.get("request_id") === requestId && pred(log),
      `the usage row for ${requestId}`,
      20_000,
    );
  }

  function expectUpstreamFailure(row: Map<string, string>, status: string): void {
    expect(row.get("status_code")).toBe(status);
    const errorClass = row.get("error_class") ?? "";
    expect(errorClass, "the row names the failure").not.toBe("");
    expect(errorClass, "an upstream failure is not the caller leaving").not.toBe("client_disconnected");
    expect(row.get("error_message") ?? "").not.toBe("");
  }

  // ── A mid-stream upstream failure, per family ──────────────────────

  test("chat/completions: a connection lost mid-stream is a 502 with its error", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const requestId = await streamToEnd("/v1/chat/completions", "sf-chat-cut");
    expectUpstreamFailure(await usageRow(requestId), "502");
  });

  test("messages (translated): a connection lost mid-stream is a 502, not the caller leaving", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const requestId = await streamToEnd("/v1/messages", "sf-messages-bridge-cut");
    expectUpstreamFailure(await usageRow(requestId), "502");
  });

  test("messages (native): an in-band rate-limit error records the 429 it would have been", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const requestId = await streamToEnd("/v1/messages", "sf-messages-native-error");
    const row = await usageRow(requestId);
    expectUpstreamFailure(row, "429");
    expect(row.get("error_message")).toContain("exceeded your rate limit");
  });

  test("responses (translated): a connection lost mid-stream is a 502 with its error", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const requestId = await streamToEnd("/v1/responses", "sf-responses-bridge-cut");
    expectUpstreamFailure(await usageRow(requestId), "502");
  });

  test("responses (native): an upstream response.failed is a 502 carrying its message", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const requestId = await streamToEnd("/v1/responses", "sf-responses-native-failed");
    const row = await usageRow(requestId);
    expectUpstreamFailure(row, "502");
    expect(row.get("error_message")).toContain("failed partway through the response");
  });

  test("responses (native, held back by an output guardrail): an upstream response.failed is still a 502", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const requestId = await streamToEnd("/v1/responses", "sf-responses-native-failed-held");
    const row = await usageRow(requestId);
    expectUpstreamFailure(row, "502");
    expect(row.get("error_message")).toContain("failed partway through the response");
    expect(row.get("guardrail_blocked") ?? "false").toBe("false");
  });

  test("responses (translated): an upstream stream that carried nothing is a 502", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const requestId = await streamToEnd("/v1/responses", "sf-responses-bridge-empty");
    const row = await usageRow(requestId);
    expectUpstreamFailure(row, "502");
    expect(row.get("error_message")).toContain("empty stream");
  });

  test("audio transcriptions: a stream that loses its upstream is a 502, not the caller leaving", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const form = new FormData();
    form.set("model", "sf-transcribe-cut");
    form.set("stream", "true");
    form.set("file", new Blob([new Uint8Array([0x49, 0x44, 0x33])], { type: "audio/mpeg" }), "a.mp3");
    const res = await fetch(`${app.proxyUrl}/v1/audio/transcriptions`, {
      method: "POST",
      headers: { authorization: `Bearer ${CALLER_PLAINTEXT}` },
      body: form,
    });
    expect(res.status).toBe(200);
    const requestId = res.headers.get("x-aisix-request-id") ?? "";
    expect(requestId).not.toBe("");
    await res.text().catch(() => undefined);
    expectUpstreamFailure(await usageRow(requestId), "502");
  });

  test("passthrough route: a stream that loses its upstream is a 502", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const res = await fetch(`${app.proxyUrl}${ROUTE_PREFIX}/v1/chat/completions`, {
      method: "POST",
      headers: { authorization: `Bearer ${CALLER_PLAINTEXT}`, "content-type": "application/json" },
      body: JSON.stringify({ model: "relay-compat-x", messages: [{ role: "user", content: "hi" }], stream: true }),
    });
    expect(res.status).toBe(200);
    const requestId = res.headers.get("x-aisix-request-id") ?? "";
    expect(requestId).not.toBe("");
    await res.text().catch(() => undefined);
    expectUpstreamFailure(await usageRow(requestId), "502");
  });

  // ── The caller leaving ─────────────────────────────────────────────

  test("responses (translated): a caller that leaves mid-stream is a 499", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const requestId = await abandonAfterFirstRead("/v1/responses", "sf-responses-bridge-trickle");
    const row = await usageRow(requestId);
    expect(row.get("status_code")).toBe("499");
    expect(row.get("error_class")).toBe("client_disconnected");
  }, 30_000);

  test("ensemble: a caller that leaves the judge's stream is a 499 on the judge's row", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const requestId = await abandonAfterFirstRead("/v1/chat/completions", ENSEMBLE);
    const judge = await usageRow(requestId, (log) => log.get("attempt_kind") === "judge");
    expect(judge.get("status_code")).toBe("499");
    expect(judge.get("error_class")).toBe("client_disconnected");
    // The panel answered in full before the judge started streaming.
    const member = await usageRow(requestId, (log) => log.get("attempt_kind") === "panel");
    expect(member.get("status_code")).toBe("200");
  }, 30_000);
});
