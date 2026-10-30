import { createHash, randomUUID } from "node:crypto";
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

// An upstream that answers HTTP 200 and reports a failure in the body.
//
// OpenRouter does this when the provider it routed to fails: a non-stream
// answer is a bare `{"error":{...}}` object, and a stream sends keep-alive
// comments and then one chunk carrying `"choices": []` beside a top-level
// `error`. Both are upstream failures: the caller is told so with the status
// the embedded code maps to, and the usage row names the failure as the
// upstream's in-band error, with the provider's own message.
//
// An `error` that reports nothing (`{}`) beside a normal answer is not a
// failure; some OpenAI-compatible servers send one on every success.
//
// Observed through the SLS exporter, which receives the usage row as the
// control plane does.

const CALLER_PLAINTEXT = "sk-in-band-error-200";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");
const CREDENTIAL_REF = "mock";
const SLS_PROJECT = "in-band-200-proj";
const LOGSTORE = "in-band-200-store";

const PROVIDER_MESSAGE = "Upstream error from Phala: the provider is unavailable";
const PROVIDER_ERROR = {
  code: 502,
  message: PROVIDER_MESSAGE,
  metadata: { error_type: "provider_unavailable" },
};

const KEEP_ALIVE = ": OPENROUTER PROCESSING\n\n";

function errorChunk(withId: boolean): string {
  const chunk: Record<string, unknown> = {
    object: "chat.completion.chunk",
    created: 1_700_000_000,
    model: "qwen/qwen-2.5-7b-instruct",
    provider: "Phala",
    choices: [],
    error: PROVIDER_ERROR,
  };
  if (withId) chunk.id = "gen-in-band";
  return `data: ${JSON.stringify(chunk)}\n\n`;
}

const ANSWER = "a plain answer";

function okChunk(delta: Record<string, unknown>, finish: string | null = null): string {
  return JSON.stringify({
    id: "chatcmpl-ok",
    object: "chat.completion.chunk",
    created: 1_700_000_000,
    model: "relay-mini",
    choices: [{ index: 0, delta, finish_reason: finish }],
    error: {},
  });
}

const EMPTY_ERROR_STREAM: OpenAiUpstreamOptions = {
  streamEvents: [okChunk({ role: "assistant", content: ANSWER }), okChunk({}, "stop"), "[DONE]"],
};

interface Seeded {
  upstream: OpenAiUpstreamOptions;
  /** `chat` and `responses-native` keys declare the upstream's own
   *  `/v1/responses`; `responses-bridge` ones do not, so that surface is
   *  translated through chat. */
  kind: "chat" | "responses-native" | "responses-bridge" | "anthropic";
}

const MODELS: Record<string, Seeded> = {
  "ib-chat-ns": { upstream: { nonStreamBody: { error: PROVIDER_ERROR } }, kind: "chat" },
  "ib-chat-stream-id": {
    upstream: { rawStreamFrames: [KEEP_ALIVE, KEEP_ALIVE, errorChunk(true)] },
    kind: "chat",
  },
  "ib-chat-stream-no-id": {
    upstream: { rawStreamFrames: [KEEP_ALIVE, KEEP_ALIVE, errorChunk(false)] },
    kind: "chat",
  },
  "ib-embed-ns": { upstream: { nonStreamBody: { error: PROVIDER_ERROR } }, kind: "chat" },
  "ib-responses-bridge-ns": {
    upstream: { nonStreamBody: { error: PROVIDER_ERROR } },
    kind: "responses-bridge",
  },
  "ib-messages-native-ns": {
    upstream: {
      nonStreamBody: {
        type: "error",
        error: { type: "overloaded_error", message: "Overloaded right now" },
      },
    },
    kind: "anthropic",
  },
  "ib-responses-native-failed-ns": {
    upstream: {
      nonStreamBody: {
        id: "resp_failed",
        object: "response",
        status: "failed",
        model: "gpt-4o",
        output: [],
        error: { code: "server_error", message: PROVIDER_MESSAGE },
        usage: { input_tokens: 7, output_tokens: 2, total_tokens: 9 },
      },
    },
    kind: "responses-native",
  },
  "ib-responses-native-bare-ns": {
    upstream: { nonStreamBody: { error: PROVIDER_ERROR } },
    kind: "responses-native",
  },
  "ib-responses-native-ok-ns": {
    upstream: {
      nonStreamBody: {
        id: "resp_ok",
        object: "response",
        status: "completed",
        model: "gpt-4o",
        output: [
          {
            type: "message",
            id: "msg_ok",
            role: "assistant",
            status: "completed",
            content: [{ type: "output_text", text: ANSWER, annotations: [] }],
          },
        ],
        error: null,
        incomplete_details: null,
        usage: { input_tokens: 5, output_tokens: 3, total_tokens: 8 },
      },
    },
    kind: "responses-native",
  },
  "ib-chat-empty-error-ns": {
    upstream: {
      nonStreamBody: {
        id: "chatcmpl-ok",
        object: "chat.completion",
        created: 1_700_000_000,
        model: "relay-mini",
        choices: [{ index: 0, message: { role: "assistant", content: ANSWER }, finish_reason: "stop" }],
        usage: { prompt_tokens: 5, completion_tokens: 3, total_tokens: 8 },
        error: {},
      },
    },
    kind: "chat",
  },
  "ib-chat-empty-error-stream": { upstream: EMPTY_ERROR_STREAM, kind: "chat" },
};

const ROUTE_NS = "ib-route-ns";
const ROUTE_NS_PREFIX = "/passthrough/ib-ns";
const ROUTE_GUARDED = "ib-route-guarded";
const ROUTE_GUARDED_PREFIX = "/passthrough/ib-guarded";
const ROUTE_EMPTY = "ib-route-empty";
const ROUTE_EMPTY_PREFIX = "/passthrough/ib-empty";

describe("an upstream 200 that reports a failure in its body", () => {
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
      name: "sls-in-band-200",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: SLS_PROJECT,
      logstore: LOGSTORE,
      credential_ref: CREDENTIAL_REF,
      content_mode: "metadata_only",
    });
    for (const [model, { upstream: opts, kind }] of Object.entries(MODELS)) {
      const upstream = await startOpenAiUpstream(opts);
      upstreams.push(upstream);
      if (kind === "anthropic") {
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
      await seed.createModel({
        display_name: model,
        provider: "openai",
        model_name: "qwen/qwen-2.5-7b-instruct",
        provider_key_id: pk.id,
      });
    }
    // An output guardrail on one route: the error body is the provider's
    // failure, not an answer for it to read.
    const outputGuard = await seed.createGuardrail(
      {
        name: "ib-output-block",
        enabled: true,
        hook_point: "output",
        kind: "keyword",
        patterns: [{ kind: "literal", value: "a phrase the upstream never says" }],
      },
      { attach: false },
    );
    for (const [name, prefix, opts] of [
      [ROUTE_NS, ROUTE_NS_PREFIX, { nonStreamBody: { error: PROVIDER_ERROR } }],
      [ROUTE_GUARDED, ROUTE_GUARDED_PREFIX, { nonStreamBody: { error: PROVIDER_ERROR } }],
      [ROUTE_EMPTY, ROUTE_EMPTY_PREFIX, EMPTY_ERROR_STREAM],
    ] as const) {
      const upstream = await startOpenAiUpstream(opts);
      upstreams.push(upstream);
      const pk = await seed.createProviderKey({
        display_name: `${name}-pk`,
        secret: "sk-mock",
        api_base: `${upstream.baseUrl}/v1`,
      });
      const route = await seed.createPassthroughRoute({
        name,
        path_prefix: prefix,
        target_url: upstream.baseUrl,
        provider_key_id: pk.id,
      });
      if (name === ROUTE_GUARDED) {
        await seed.update("guardrail_attachments", randomUUID(), {
          guardrail_id: outputGuard.id,
          scope_type: "passthrough_route",
          scope_id: route.id,
          priority: 100,
        });
      }
    }
    // Seeded last: the key authenticating implies the whole seed set is in
    // the snapshot.
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: Object.keys(MODELS),
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

  async function post(path: string, body: Record<string, unknown>): Promise<{
    status: number;
    requestId: string;
    text: string;
  }> {
    const res = await fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: { authorization: `Bearer ${CALLER_PLAINTEXT}`, "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    const requestId = res.headers.get("x-aisix-request-id") ?? "";
    expect(requestId).not.toBe("");
    return { status: res.status, requestId, text: await res.text() };
  }

  async function usageRow(requestId: string): Promise<Map<string, string>> {
    return waitForSlsLog(
      sls!,
      LOGSTORE,
      (log) => log.get("request_id") === requestId,
      `the usage row for ${requestId}`,
      20_000,
    );
  }

  function expectInBandFailure(row: Map<string, string>, status: string, message: string): void {
    expect(row.get("status_code")).toBe(status);
    expect(row.get("error_class")).toBe("upstream_in_band");
    expect(row.get("error_message")).toContain(message);
  }

  function expectClean(row: Map<string, string>): void {
    expect(row.get("status_code")).toBe("200");
    expect(row.get("error_class") ?? "").toBe("");
  }

  const chat = (model: string, stream: boolean) => ({
    model,
    messages: [{ role: "user", content: "hi" }],
    stream,
  });

  test("chat/completions: a bare error body is a 502 carrying the provider's message", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const res = await post("/v1/chat/completions", chat("ib-chat-ns", false));
    expect(res.status).toBe(502);
    expect(res.text).toContain(PROVIDER_MESSAGE);
    expectInBandFailure(await usageRow(res.requestId), "502", PROVIDER_MESSAGE);
  });

  for (const model of ["ib-chat-stream-id", "ib-chat-stream-no-id"]) {
    test(`chat/completions stream (${model}): an error chunk with empty choices is a failure`, async (ctx) => {
      if (!etcdReachable || !app || !sls) return ctx.skip();
      const res = await post("/v1/chat/completions", chat(model, true));
      // The stream's head is already out; the caller reads the failure as
      // the stream's error event.
      expect(res.status).toBe(200);
      expect(res.text).toContain(PROVIDER_MESSAGE);
      expectInBandFailure(await usageRow(res.requestId), "502", PROVIDER_MESSAGE);
    });
  }

  test("embeddings: a bare error body is a 502 carrying the provider's message", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const res = await post("/v1/embeddings", { model: "ib-embed-ns", input: "hi" });
    expect(res.status).toBe(502);
    expect(res.text).toContain(PROVIDER_MESSAGE);
    expectInBandFailure(await usageRow(res.requestId), "502", PROVIDER_MESSAGE);
  });

  test("responses (translated): a bare error body is a 502 carrying the provider's message", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const res = await post("/v1/responses", { model: "ib-responses-bridge-ns", input: "hi" });
    expect(res.status).toBe(502);
    expect(res.text).toContain(PROVIDER_MESSAGE);
    expectInBandFailure(await usageRow(res.requestId), "502", PROVIDER_MESSAGE);
  });

  test("messages (native): an error body is a failure carrying the provider's message", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const res = await post("/v1/messages", {
      model: "ib-messages-native-ns",
      max_tokens: 64,
      messages: [{ role: "user", content: "hi" }],
    });
    expect(res.status).toBe(502);
    expect(res.text).toContain("Overloaded right now");
    expectInBandFailure(await usageRow(res.requestId), "502", "Overloaded right now");
  });

  test("chat/completions onto an Anthropic upstream: an error body is a failure", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const res = await post("/v1/chat/completions", chat("ib-messages-native-ns", false));
    expect(res.status).toBe(502);
    expect(res.text).toContain("Overloaded right now");
    expectInBandFailure(await usageRow(res.requestId), "502", "Overloaded right now");
  });

  test("messages/count_tokens: an error body is a failure, not a count", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const res = await post("/v1/messages/count_tokens", {
      model: "ib-messages-native-ns",
      messages: [{ role: "user", content: "hi" }],
    });
    expect(res.status).toBe(502);
    expect(res.text).toContain("Overloaded right now");
  });

  test("responses (native): a bare error body is a 502 carrying the provider's message", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const res = await post("/v1/responses", { model: "ib-responses-native-bare-ns", input: "hi" });
    expect(res.status).toBe(502);
    expect(res.text).toContain(PROVIDER_MESSAGE);
    expectInBandFailure(await usageRow(res.requestId), "502", PROVIDER_MESSAGE);
  });

  test("responses (native): a failed response object is relayed as the answer, with its usage", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const res = await post("/v1/responses", { model: "ib-responses-native-failed-ns", input: "hi" });
    expect(res.status).toBe(200);
    const body = JSON.parse(res.text);
    expect(body.status).toBe("failed");
    expect(body.error.message).toBe(PROVIDER_MESSAGE);
    const row = await usageRow(res.requestId);
    expectClean(row);
    expect(row.get("prompt_tokens")).toBe("7");
    expect(row.get("completion_tokens")).toBe("2");
  });

  test("responses (native): a completed response with a null error is a success", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const res = await post("/v1/responses", { model: "ib-responses-native-ok-ns", input: "hi" });
    expect(res.status).toBe(200);
    expect(res.text).toContain(ANSWER);
    expectClean(await usageRow(res.requestId));
  });

  test("passthrough route: a relayed error body records the failure", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const res = await post(`${ROUTE_NS_PREFIX}/v1/chat/completions`, chat("qwen/qwen-2.5-7b-instruct", false));
    // The route relays the upstream's answer as it came.
    expect(res.status).toBe(200);
    expect(JSON.parse(res.text)).toEqual({ error: PROVIDER_ERROR });
    expectInBandFailure(await usageRow(res.requestId), "502", PROVIDER_MESSAGE);
  });

  test("passthrough route behind an output guardrail: the error body is the provider's failure", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const res = await post(`${ROUTE_GUARDED_PREFIX}/v1/chat/completions`, chat("qwen/qwen-2.5-7b-instruct", false));
    expect(res.status).toBe(200);
    expect(JSON.parse(res.text)).toEqual({ error: PROVIDER_ERROR });
    const row = await usageRow(res.requestId);
    expectInBandFailure(row, "502", PROVIDER_MESSAGE);
    expect(row.get("guardrail_blocked") ?? "false").toBe("false");
  });

  // ── An `error` that reports nothing ────────────────────────────────

  test("chat/completions: an empty error beside an answer is a success", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const res = await post("/v1/chat/completions", chat("ib-chat-empty-error-ns", false));
    expect(res.status).toBe(200);
    expect(JSON.parse(res.text).choices[0].message.content).toBe(ANSWER);
    expectClean(await usageRow(res.requestId));
  });

  test("chat/completions stream: an empty error on every chunk is a success", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const res = await post("/v1/chat/completions", chat("ib-chat-empty-error-stream", true));
    expect(res.status).toBe(200);
    expect(res.text).toContain(ANSWER);
    expectClean(await usageRow(res.requestId));
  });

  test("passthrough route stream: an empty error on every chunk is a success", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const res = await post(`${ROUTE_EMPTY_PREFIX}/v1/chat/completions`, chat("relay-mini", true));
    expect(res.status).toBe(200);
    expect(res.text).toContain(ANSWER);
    expectClean(await usageRow(res.requestId));
  });
});
