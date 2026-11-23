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

// An upstream that reports usage but leaves the prompt count at zero (or
// leaves it out) while still reporting a total: `{completion_tokens: N,
// total_tokens: N}`. The gateway estimates the prompt, so the upstream's
// total no longer adds up to the counts the usage record carries.
//
// Contract pinned here, on every inbound protocol and both response modes:
// the usage record carries `usage_estimated: true` and NO `total_tokens` —
// the control plane derives a total for a record without one. Where the
// endpoint hands the client the estimate, the client's counters are the
// record's and its total adds up: estimated prompt + N.
//
// The record is read off a metadata-only SLS exporter, which serializes the
// usage event field for field (a field the event omits is absent there).

const CALLER_PLAINTEXT = "sk-estimated-total-caller";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");

const CREDENTIAL_REF = "e2e";
const SLS_PROJECT = "estimated-total-proj";
const LOGSTORE = "estimated-total-store";

const OUT = 2;

// Chat Completions shape: prompt_tokens absent on the buffered reply, a
// reported 0 on the stream.
const CHAT_BODY = {
  id: "chatcmpl-est-total",
  object: "chat.completion",
  created: 1_700_000_000,
  model: "relay-compat-x",
  choices: [
    { index: 0, message: { role: "assistant", content: "Hello world" }, finish_reason: "stop" },
  ],
  usage: { completion_tokens: OUT, total_tokens: OUT },
};
const CHAT_STREAM = [
  '{"id":"chatcmpl-est-total","object":"chat.completion.chunk","model":"relay-compat-x","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}',
  '{"id":"chatcmpl-est-total","object":"chat.completion.chunk","model":"relay-compat-x","choices":[{"index":0,"delta":{"content":"Hello world"},"finish_reason":null}]}',
  '{"id":"chatcmpl-est-total","object":"chat.completion.chunk","model":"relay-compat-x","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
  `{"id":"chatcmpl-est-total","object":"chat.completion.chunk","model":"relay-compat-x","choices":[],"usage":{"prompt_tokens":0,"completion_tokens":${OUT},"total_tokens":${OUT}}}`,
  "[DONE]",
];

// Responses shape, served natively.
const RESPONSE_OBJECT = {
  id: "resp_est_total",
  object: "response",
  created_at: 1_700_000_000,
  model: "relay-compat-x",
  status: "completed",
  output: [
    {
      type: "message",
      id: "msg_est_total",
      role: "assistant",
      status: "completed",
      content: [{ type: "output_text", text: "Hello world", annotations: [] }],
    },
  ],
  usage: { input_tokens: 0, output_tokens: OUT, total_tokens: OUT },
};
const RESPONSES_STREAM = [
  JSON.stringify({
    type: "response.created",
    response: { ...RESPONSE_OBJECT, status: "in_progress", output: [], usage: null },
  }),
  JSON.stringify({ type: "response.output_text.delta", item_id: "msg_est_total", output_index: 0, content_index: 0, delta: "Hello world" }),
  JSON.stringify({ type: "response.completed", response: RESPONSE_OBJECT }),
];

// Legacy Completions shape.
const COMPLETIONS_BODY = {
  id: "cmpl-est-total",
  object: "text_completion",
  created: 1_700_000_000,
  model: "relay-compat-x",
  choices: [{ index: 0, text: "Hello world", finish_reason: "stop" }],
  usage: { prompt_tokens: 0, completion_tokens: OUT, total_tokens: OUT },
};

type Upstreams = {
  chat: OpenAiUpstream;
  chatStream: OpenAiUpstream;
  responses: OpenAiUpstream;
  responsesStream: OpenAiUpstream;
  completions: OpenAiUpstream;
};

/** Model display name → the upstream it is served from and whether its
 * provider key declares no Responses surface (so `/v1/responses` bridges
 * to Chat Completions). */
const MODELS: Record<string, { upstream: keyof Upstreams; bridge?: boolean }> = {
  "est-total-chat": { upstream: "chat" },
  "est-total-chat-stream": { upstream: "chatStream" },
  "est-total-responses": { upstream: "responses" },
  "est-total-responses-stream": { upstream: "responsesStream" },
  "est-total-responses-bridge": { upstream: "chat", bridge: true },
  "est-total-responses-bridge-stream": { upstream: "chatStream", bridge: true },
  "est-total-completions": { upstream: "completions" },
};

function lastSseUsage(
  text: string,
  pick: (frame: Record<string, unknown>) => unknown,
): Record<string, number> {
  let found: Record<string, number> | undefined;
  for (const line of text.split("\n")) {
    if (!line.startsWith("data: ")) continue;
    const data = line.slice(6).trim();
    if (data === "[DONE]") continue;
    let frame: Record<string, unknown>;
    try {
      frame = JSON.parse(data) as Record<string, unknown>;
    } catch {
      continue;
    }
    const usage = pick(frame);
    if (usage && typeof usage === "object") found = usage as Record<string, number>;
  }
  if (!found) throw new Error(`no usage in stream:\n${text}`);
  return found;
}

describe("a usage record with an estimated counter carries no total", () => {
  let etcdReachable = false;
  let app: SpawnedApp | undefined;
  let sls: MockSls | undefined;
  let upstreams: Upstreams | undefined;

  async function call(path: string, body: unknown): Promise<{ requestId: string; text: string }> {
    const res = await fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
      },
      body: JSON.stringify(body),
    });
    const text = await res.text();
    expect(res.status, text).toBe(200);
    const requestId = res.headers.get("x-aisix-request-id") ?? "";
    expect(requestId).not.toBe("");
    return { requestId, text };
  }

  /** The request's usage record: estimated, upstream's completion kept, no
   * total. Returns its prompt count. */
  async function expectRecordWithoutTotal(requestId: string): Promise<number> {
    const log = await waitForSlsLog(
      sls!,
      LOGSTORE,
      (l) => l.get("request_id") === requestId,
      `usage row for ${requestId}`,
      15_000,
    );
    expect(log.get("usage_estimated")).toBe("true");
    expect(log.get("completion_tokens")).toBe(String(OUT));
    const prompt = Number(log.get("prompt_tokens") ?? "0");
    expect(prompt).toBeGreaterThan(0);
    expect(log.get("total_tokens")).toBeUndefined();
    return prompt;
  }

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    sls = await startMockSls();
    upstreams = {
      chat: await startOpenAiUpstream({ nonStreamBody: CHAT_BODY }),
      chatStream: await startOpenAiUpstream({ streamEvents: CHAT_STREAM }),
      responses: await startOpenAiUpstream({ nonStreamBody: RESPONSE_OBJECT }),
      responsesStream: await startOpenAiUpstream({ streamEvents: RESPONSES_STREAM }),
      completions: await startOpenAiUpstream({ nonStreamBody: COMPLETIONS_BODY }),
    };

    app = await spawnApp({
      extraEnv: {
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: "mock-ak-id",
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: "mock-ak-secret",
      },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createObservabilityExporter({
      name: "estimated-total-sls",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: SLS_PROJECT,
      logstore: LOGSTORE,
      credential_ref: CREDENTIAL_REF,
      content_mode: "metadata_only",
    });

    // Non-OpenAI upstream model names, so the estimator's fallback
    // encoding applies.
    for (const [display, { upstream, bridge }] of Object.entries(MODELS)) {
      const pk = await seed.createProviderKey({
        display_name: `${display}-pk`,
        api_key: "sk-mock",
        api_base: `${upstreams[upstream].baseUrl}/v1`,
        ...(bridge ? { apis: {} } : {}),
      });
      await seed.createModel({
        display_name: display,
        provider: "openai",
        model_name: "relay-compat-x",
        provider_key_id: pk.id,
      });
    }

    // Caller key last: it authenticating implies the whole seed set is in
    // the snapshot (tests/e2e/AGENTS.md).
    await seed.createApiKey({ key_hash: CALLER_KEY_HASH, allowed_models: Object.keys(MODELS) });
    const probe = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    await waitConfigPropagation(async () => (await probe.listModels()).status === 200);
  });

  afterAll(async () => {
    await app?.exit();
    await sls?.close();
    for (const u of Object.values(upstreams ?? {})) await u.close();
  });

  test("/v1/chat/completions, buffered, prompt_tokens absent", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const res = await call("/v1/chat/completions", {
      model: "est-total-chat",
      messages: [{ role: "user", content: "hi" }],
    });
    const usage = (JSON.parse(res.text) as { usage: Record<string, number> }).usage;
    const prompt = await expectRecordWithoutTotal(res.requestId);
    expect(usage.prompt_tokens).toBe(prompt);
    expect(usage.completion_tokens).toBe(OUT);
    expect(usage.total_tokens).toBe(prompt + OUT);
  });

  test("/v1/chat/completions, streamed with include_usage, prompt_tokens 0", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const res = await call("/v1/chat/completions", {
      model: "est-total-chat-stream",
      messages: [{ role: "user", content: "hi" }],
      stream: true,
      stream_options: { include_usage: true },
    });
    const usage = lastSseUsage(res.text, (f) => f.usage);
    const prompt = await expectRecordWithoutTotal(res.requestId);
    expect(usage.prompt_tokens).toBe(prompt);
    expect(usage.completion_tokens).toBe(OUT);
    expect(usage.total_tokens).toBe(prompt + OUT);
  });

  test("/v1/messages over a Chat Completions upstream, buffered", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const res = await call("/v1/messages", {
      model: "est-total-chat",
      max_tokens: 16,
      messages: [{ role: "user", content: "hi" }],
    });
    const usage = (JSON.parse(res.text) as { usage: Record<string, number> }).usage;
    const prompt = await expectRecordWithoutTotal(res.requestId);
    expect(usage.input_tokens).toBe(prompt);
    expect(usage.output_tokens).toBe(OUT);
  });

  test("/v1/messages over a Chat Completions upstream, streamed", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const res = await call("/v1/messages", {
      model: "est-total-chat-stream",
      max_tokens: 16,
      stream: true,
      messages: [{ role: "user", content: "hi" }],
    });
    await expectRecordWithoutTotal(res.requestId);
  });

  test("/v1/responses, native, buffered, input_tokens 0", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const res = await call("/v1/responses", { model: "est-total-responses", input: "hi" });
    await expectRecordWithoutTotal(res.requestId);
  });

  test("/v1/responses, native, streamed, input_tokens 0", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const res = await call("/v1/responses", {
      model: "est-total-responses-stream",
      input: "hi",
      stream: true,
    });
    await expectRecordWithoutTotal(res.requestId);
  });

  test("/v1/responses bridged to Chat Completions, buffered", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const res = await call("/v1/responses", { model: "est-total-responses-bridge", input: "hi" });
    const usage = (JSON.parse(res.text) as { usage: Record<string, number> }).usage;
    const prompt = await expectRecordWithoutTotal(res.requestId);
    expect(usage.input_tokens).toBe(prompt);
    expect(usage.output_tokens).toBe(OUT);
    expect(usage.total_tokens).toBe(prompt + OUT);
  });

  test("/v1/responses bridged to Chat Completions, streamed", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const res = await call("/v1/responses", {
      model: "est-total-responses-bridge-stream",
      input: "hi",
      stream: true,
    });
    await expectRecordWithoutTotal(res.requestId);
  });

  test("/v1/completions, prompt_tokens 0", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const res = await call("/v1/completions", { model: "est-total-completions", prompt: "hi" });
    await expectRecordWithoutTotal(res.requestId);
  });
});
